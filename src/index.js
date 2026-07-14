import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import cors from "cors";
import express from "express";
import rateLimit from "express-rate-limit";
import { OAuth2Client } from "google-auth-library";
import helmet from "helmet";
import jwt from "jsonwebtoken";
import multer from "multer";
import OpenAI from "openai";

const {
  ALLOWED_WORKSPACE_DOMAIN = "whatstheweather.tv",
  APP_CALLBACK_URL = "whatthenote://auth/callback",
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
  OPENAI_API_KEY,
  PUBLIC_BASE_URL = "http://localhost:3000",
  SESSION_SECRET
} = process.env;

const requiredEnv = {
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
  OPENAI_API_KEY,
  SESSION_SECRET
};

for (const [key, value] of Object.entries(requiredEnv)) {
  if (!value) {
    console.warn(`[startup] Missing environment variable: ${key}`);
  }
}

const app = express();
const port = Number(process.env.PORT || 3000);
const upload = multer({
  dest: path.join(os.tmpdir(), "what-the-note-uploads"),
  limits: { fileSize: 1024 * 1024 * 1024 }
});
const oauth = new OAuth2Client(
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
  `${PUBLIC_BASE_URL}/auth/google/callback`
);
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

app.use(helmet());
app.use(cors({ origin: false }));
app.use(express.json({ limit: "10mb" }));
app.use(rateLimit({ windowMs: 60_000, limit: 120 }));

const usageByUser = new Map();

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "what-the-note-server" });
});

app.get("/auth/google/start", (req, res) => {
  const requestedRedirect = String(req.query.redirect_uri || APP_CALLBACK_URL);
  const redirectToApp = requestedRedirect === APP_CALLBACK_URL ? requestedRedirect : APP_CALLBACK_URL;
  const state = signState({ redirectToApp });
  const url = oauth.generateAuthUrl({
    access_type: "offline",
    prompt: "select_account",
    scope: ["openid", "email", "profile"],
    hd: ALLOWED_WORKSPACE_DOMAIN,
    state
  });
  res.redirect(url);
});

app.get("/auth/google/callback", async (req, res) => {
  try {
    const code = String(req.query.code || "");
    const state = verifyState(String(req.query.state || ""));
    if (!code || !state?.redirectToApp) {
      return redirectToAppError(res, APP_CALLBACK_URL, "missing_login_code");
    }

    const { tokens } = await oauth.getToken(code);
    const ticket = await oauth.verifyIdToken({
      idToken: tokens.id_token,
      audience: GOOGLE_CLIENT_ID
    });
    const user = userFromPayload(ticket.getPayload());
    ensureAllowedUser(user);

    const appToken = issueSessionToken(user);
    const callback = new URL(state.redirectToApp);
    callback.searchParams.set("token", appToken);
    res.redirect(callback.toString());
  } catch (error) {
    console.error("[auth callback]", error);
    redirectToAppError(res, APP_CALLBACK_URL, "auth_failed");
  }
});

app.post("/auth/google/verify", async (req, res) => {
  try {
    const expected = String(req.body.expectedWorkspaceDomain || ALLOWED_WORKSPACE_DOMAIN).toLowerCase();
    const decoded = verifySessionToken(String(req.body.token || ""));
    if (decoded.workspaceDomain !== expected) {
      return res.status(403).json({ message: "허용되지 않은 Google Workspace 계정입니다." });
    }
    res.json({
      id: decoded.sub,
      email: decoded.email,
      workspaceDomain: decoded.workspaceDomain,
      sessionToken: req.body.token
    });
  } catch (error) {
    res.status(401).json({ message: "로그인 세션이 만료되었습니다." });
  }
});

app.get("/usage/me", requireSession, (req, res) => {
  const usage = usageFor(req.user.email);
  res.json({
    transcriptionMinutesUsed: usage.transcriptionMinutesUsed,
    transcriptionMinutesLimit: Number(process.env.TRANSCRIPTION_MINUTES_LIMIT || 1000),
    subtitleProjectsUsed: usage.subtitleProjectsUsed,
    subtitleProjectsLimit: Number(process.env.SUBTITLE_PROJECTS_LIMIT || 200)
  });
});

app.post("/transcribe", requireSession, upload.single("file"), async (req, res) => {
  try {
    const filePath = req.file?.path || req.body.compressedAudioPath || req.body.sourceFilePath;
    if (!filePath || !fs.existsSync(filePath)) {
      return res.status(400).json({ message: "전사할 파일을 찾지 못했습니다." });
    }

    const model = process.env.OPENAI_TRANSCRIPTION_MODEL || "gpt-4o-transcribe";
    const result = await tryTranscription({
      filePath,
      language: req.body.language || "ko",
      model
    });

    incrementUsage(req.user.email, { transcriptionMinutesUsed: 1 });
    res.json({
      transcriptText: result.text || "",
      segments: []
    });
  } catch (error) {
    console.error("[transcribe]", error);
    const message = describeOpenAIError(error);
    if (message.includes("maximum") || message.includes("larger than") || message.includes("file size")) {
      return res.status(413).json({ message: "전사 파일이 너무 큽니다. 앱에서 더 작게 압축한 뒤 다시 시도해 주세요." });
    }
    res.status(error?.status || 500).json({ message: `전사 처리에 실패했습니다.\n${message}` });
  } finally {
    cleanupUpload(req.file?.path);
  }
});

app.post("/summarize", requireSession, async (req, res) => {
  try {
    const transcriptText = String(req.body.transcriptText || "");
    const terms = req.body.terms || [];
    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_SUMMARY_MODEL || "gpt-4.1-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "You turn Korean meeting transcripts into concise notes, action items, and execution checklists. Return strict JSON."
        },
        {
          role: "user",
          content: JSON.stringify({
            transcriptText,
            confirmedTerms: terms,
            schema: {
              summary: "string",
              actionItems: [{ title: "string", owner: "string", dueDateText: "string", isDone: false }],
              checklist: [{ title: "string", owner: "string", dueDateText: "string", isDone: false }]
            }
          })
        }
      ]
    });
    const parsed = parseJSON(completion.choices[0]?.message?.content);
    res.json({
      summary: parsed.summary || "",
      actionItems: normalizeChecklist(parsed.actionItems),
      checklist: normalizeChecklist(parsed.checklist)
    });
  } catch (error) {
    console.error("[summarize]", error);
    res.status(500).json({ message: "회의 정리 생성에 실패했습니다." });
  }
});

app.post("/term-candidates", requireSession, async (req, res) => {
  try {
    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_SUMMARY_MODEL || "gpt-4.1-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "Find only proper nouns, names, brands, projects, acronyms, and suspicious transcription terms. Exclude everyday filler words. Return strict JSON."
        },
        {
          role: "user",
          content: JSON.stringify({
            text: req.body.text || "",
            existingTerms: req.body.existingTerms || [],
            schema: {
              terms: [
                {
                  original: "string",
                  confirmedSourceText: "string",
                  targetText: "string",
                  type: "person|brand|project|product|place|term|acronym|unknown",
                  note: "string",
                  isConfirmed: false
                }
              ]
            }
          })
        }
      ]
    });
    const parsed = parseJSON(completion.choices[0]?.message?.content);
    res.json({ terms: normalizeTerms(parsed.terms) });
  } catch (error) {
    console.error("[term-candidates]", error);
    res.status(500).json({ message: "AI 용어 후보 찾기에 실패했습니다." });
  }
});

app.post("/subtitle/analyze", requireSession, (_req, res) => {
  res.json({
    duration: 0,
    sceneCuts: [],
    speechSegments: [],
    audioFileHint: null
  });
});

app.post("/subtitle/generate", requireSession, upload.single("file"), async (req, res) => {
  try {
    const filePath = req.file?.path;
    if (!filePath || !fs.existsSync(filePath)) {
      return res.status(400).json({ message: "자막을 만들 오디오 파일을 찾지 못했습니다." });
    }

    const sourceLanguage = String(req.body.sourceLanguage || "ko");
    const sceneCuts = parseJSONArray(req.body.sceneCuts);
    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(filePath),
      model: process.env.OPENAI_SUBTITLE_TRANSCRIPTION_MODEL || "whisper-1",
      language: sourceLanguage,
      response_format: "verbose_json",
      timestamp_granularities: ["segment"]
    });
    const transcriptSegments = normalizeTranscriptionSegments(transcription.segments);

    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_SUMMARY_MODEL || "gpt-4.1-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "Create readable subtitles from timestamped transcript segments. Preserve spoken meaning, keep timing monotonic, and split at natural meaning units. Use nearby scene cuts when they improve readability. Return strict JSON."
        },
        {
          role: "user",
          content: JSON.stringify({
            sourceLanguage,
            sceneCuts,
            transcriptSegments,
            schema: {
              segments: [{ index: 1, startTime: 0, endTime: 2, text: "string", translatedText: "" }],
              glossaryTerms: []
            }
          })
        }
      ]
    });
    const parsed = parseJSON(completion.choices[0]?.message?.content);
    const segments = normalizeSubtitleSegments(parsed.segments).length
      ? normalizeSubtitleSegments(parsed.segments)
      : normalizeSubtitleSegments(transcriptSegments);
    incrementUsage(req.user.email, { subtitleProjectsUsed: 1 });
    res.json({
      originalSRT: buildSRT(segments, "text"),
      segments,
      glossaryTerms: normalizeTerms(parsed.glossaryTerms)
    });
  } catch (error) {
    console.error("[subtitle generate]", error);
    res.status(500).json({ message: "자막 생성에 실패했습니다." });
  } finally {
    cleanupUpload(req.file?.path);
  }
});

app.post("/subtitle/translate", requireSession, async (req, res) => {
  try {
    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_SUMMARY_MODEL || "gpt-4.1-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "Translate subtitles naturally for video. Preserve confirmed glossary terms and keep SRT readable. Return strict JSON."
        },
        {
          role: "user",
          content: JSON.stringify({
            sourceLanguage: req.body.sourceLanguage || "ko",
            targetLanguage: req.body.targetLanguage || "en",
            originalSRT: req.body.originalSRT || "",
            terms: req.body.terms || [],
            schema: {
              translatedSRT: "string",
              segments: [{ index: 1, startTime: 0, endTime: 2, text: "string", translatedText: "string" }],
              glossaryTerms: []
            }
          })
        }
      ]
    });
    const parsed = parseJSON(completion.choices[0]?.message?.content);
    const segments = normalizeSubtitleSegments(parsed.segments);
    res.json({
      translatedSRT: parsed.translatedSRT || buildSRT(segments, "translatedText"),
      segments,
      glossaryTerms: normalizeTerms(parsed.glossaryTerms)
    });
  } catch (error) {
    console.error("[subtitle translate]", error);
    res.status(500).json({ message: "자막 통번역에 실패했습니다." });
  }
});

app.listen(port, () => {
  console.log(`what the note server listening on :${port}`);
});

function requireSession(req, res, next) {
  try {
    const header = req.headers.authorization || "";
    const token = header.startsWith("Bearer ") ? header.slice(7) : "";
    req.user = verifySessionToken(token);
    next();
  } catch {
    res.status(401).json({ message: "로그인 세션이 만료되었습니다." });
  }
}

function userFromPayload(payload = {}) {
  const email = String(payload.email || "").toLowerCase();
  return {
    id: String(payload.sub || ""),
    email,
    workspaceDomain: String(payload.hd || email.split("@")[1] || "").toLowerCase()
  };
}

function ensureAllowedUser(user) {
  if (!user.email || !user.id) {
    throw new Error("missing_google_user");
  }
  if (user.workspaceDomain !== ALLOWED_WORKSPACE_DOMAIN) {
    throw new Error("workspace_not_allowed");
  }
}

function issueSessionToken(user) {
  return jwt.sign(
    {
      email: user.email,
      workspaceDomain: user.workspaceDomain
    },
    SESSION_SECRET,
    {
      subject: user.id,
      expiresIn: "12h",
      issuer: "what-the-note-server"
    }
  );
}

function verifySessionToken(token) {
  return jwt.verify(token, SESSION_SECRET, {
    issuer: "what-the-note-server"
  });
}

function signState(payload) {
  return jwt.sign(payload, SESSION_SECRET, { expiresIn: "10m" });
}

function verifyState(token) {
  return jwt.verify(token, SESSION_SECRET);
}

function redirectToAppError(res, fallbackURL, errorCode) {
  const callback = new URL(fallbackURL);
  callback.searchParams.set("error", errorCode);
  res.redirect(callback.toString());
}

function parseJSON(content) {
  if (!content) return {};
  try {
    return JSON.parse(content.replace(/^```json\s*/i, "").replace(/```$/i, ""));
  } catch {
    return {};
  }
}

function parseJSONArray(value) {
  if (Array.isArray(value)) return value;
  try {
    const parsed = JSON.parse(String(value || "[]"));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function normalizeChecklist(items) {
  if (!Array.isArray(items)) return [];
  return items.map((item) => ({
    id: item.id || crypto.randomUUID(),
    title: String(item.title || ""),
    owner: String(item.owner || ""),
    dueDateText: String(item.dueDateText || ""),
    isDone: Boolean(item.isDone)
  }));
}

function normalizeTerms(terms) {
  if (!Array.isArray(terms)) return [];
  return terms.map((term) => ({
    id: term.id || crypto.randomUUID(),
    original: String(term.original || ""),
    confirmedSourceText: String(term.confirmedSourceText || ""),
    targetText: String(term.targetText || ""),
    type: String(term.type || "unknown"),
    note: String(term.note || ""),
    isConfirmed: Boolean(term.isConfirmed)
  }));
}

function normalizeSubtitleSegments(segments) {
  if (!Array.isArray(segments)) return [];
  return segments.map((segment, index) => ({
    id: segment.id || crypto.randomUUID(),
    index: Number(segment.index || index + 1),
    startTime: Number(segment.startTime || 0),
    endTime: Number(segment.endTime || 0),
    text: String(segment.text || ""),
    translatedText: String(segment.translatedText || "")
  }));
}

function normalizeTranscriptionSegments(segments) {
  if (!Array.isArray(segments)) return [];
  return segments.map((segment, index) => ({
    index: index + 1,
    startTime: Number(segment.start || 0),
    endTime: Number(segment.end || 0),
    text: String(segment.text || "").trim(),
    translatedText: ""
  })).filter((segment) => segment.text && segment.endTime > segment.startTime);
}

async function tryTranscription({ filePath, language, model }) {
  return openai.audio.transcriptions.create({
    file: fs.createReadStream(filePath),
    model,
    language,
    response_format: "json"
  });
}

function describeOpenAIError(error) {
  const parts = [
    error?.name ? `name=${error.name}` : "",
    error?.status ? `status=${error.status}` : "",
    error?.code ? `code=${error.code}` : "",
    error?.type ? `type=${error.type}` : "",
    error?.param ? `param=${error.param}` : "",
    error?.request_id ? `request_id=${error.request_id}` : "",
    error?.message ? String(error.message) : ""
  ].filter(Boolean);
  return parts.join(" / ") || "알 수 없는 오류";
}

function buildSRT(segments, textField) {
  return segments.map((segment, index) => [
    index + 1,
    `${srtTimecode(segment.startTime)} --> ${srtTimecode(segment.endTime)}`,
    String(segment[textField] || "").trim()
  ].join("\n")).join("\n\n");
}

function srtTimecode(seconds) {
  const milliseconds = Math.max(0, Math.round(Number(seconds || 0) * 1000));
  const hours = Math.floor(milliseconds / 3_600_000);
  const minutes = Math.floor((milliseconds % 3_600_000) / 60_000);
  const secs = Math.floor((milliseconds % 60_000) / 1000);
  const millis = milliseconds % 1000;
  return `${pad(hours, 2)}:${pad(minutes, 2)}:${pad(secs, 2)},${pad(millis, 3)}`;
}

function pad(value, width) {
  return String(value).padStart(width, "0");
}

function usageFor(email) {
  if (!usageByUser.has(email)) {
    usageByUser.set(email, {
      transcriptionMinutesUsed: 0,
      subtitleProjectsUsed: 0
    });
  }
  return usageByUser.get(email);
}

function incrementUsage(email, patch) {
  const usage = usageFor(email);
  usage.transcriptionMinutesUsed += patch.transcriptionMinutesUsed || 0;
  usage.subtitleProjectsUsed += patch.subtitleProjectsUsed || 0;
}

function cleanupUpload(filePath) {
  if (filePath) {
    fs.promises.unlink(filePath).catch(() => {});
  }
}
