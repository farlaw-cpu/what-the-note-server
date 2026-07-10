# what the note server

This is the minimum team server for the macOS app.

Users only see Google login. The server verifies Google Workspace membership, stores the OpenAI API key in environment variables, and proxies AI requests.

## Local Test

```bash
cd Server
npm install
cp .env.example .env
npm run dev
```

For local testing, set `PUBLIC_BASE_URL=http://localhost:3000` in `.env`.

## Render Settings

Create a Render Web Service with:

```text
Runtime: Node
Build Command: pnpm install --frozen-lockfile
Start Command: npm start
```

Environment variables:

```text
ALLOWED_WORKSPACE_DOMAIN=whatstheweather.tv
APP_CALLBACK_URL=whatthenote://auth/callback
GOOGLE_CLIENT_ID=from Google Cloud
GOOGLE_CLIENT_SECRET=from Google Cloud
OPENAI_API_KEY=from OpenAI
PUBLIC_BASE_URL=https://api.whatstheweather.tv
SESSION_SECRET=long random string
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe
OPENAI_SUBTITLE_TRANSCRIPTION_MODEL=whisper-1
OPENAI_SUMMARY_MODEL=gpt-4.1-mini
```

The repository also includes `render.yaml`. Render Blueprint deploys can read the
non-secret values automatically and prompt for the Google and OpenAI credentials.

## Google OAuth

Google Cloud OAuth Client type:

```text
Web application
```

Authorized redirect URI:

```text
https://api.whatstheweather.tv/auth/google/callback
```

## Login Flow

1. App opens `/auth/google/start`.
2. Server sends user to Google login.
3. Google returns to `/auth/google/callback`.
4. Server checks `whatstheweather.tv`.
5. Server redirects back to the macOS app:

```text
whatthenote://auth/callback?token=...
```

6. App verifies that token at `/auth/google/verify`.
