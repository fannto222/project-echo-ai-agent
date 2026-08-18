# Project Echo — Zero-Budget YouTube AI Agent V1

A cloud-first, zero-paid-API starter for an English faceless storytelling channel.

## What V1 does

- Generates an original Short or long-form story with Gemini free tier.
- Generates narration with Gemini Flash TTS free tier.
- Retrieves licensed/free visual footage through the Pexels API.
- Renders video and burned-in subtitles with FFmpeg.
- Creates a thumbnail.
- Builds a rights ledger for every run.
- Uploads to YouTube via the official YouTube Data API.
- Sets `containsSyntheticMedia` when the content plan says realistic synthetic media is present.
- Keeps topic memory to reduce repetition.
- Takes a daily analytics snapshot.
- Hard-blocks paid APIs and paid ads in the zero-budget build.

## What V1 intentionally does NOT do

- It does not buy views, likes, subscribers, comments, or watch time.
- It does not use browser bots to post to TikTok.
- It does not use copyrighted songs or ripped clips.
- It does not run paid advertising.
- It does not auto-publish TikTok until an approved TikTok publishing route is connected.
- It defaults YouTube uploads to **private** because new unverified YouTube API projects are restricted to private uploads until Google's compliance audit is completed.

## Free services used

1. **GitHub Actions** — cloud execution. Standard runners are free for public repositories. Private GitHub Free repositories have a monthly included-minutes quota.
2. **Gemini Developer API free tier** — script generation and Flash TTS.
3. **Pexels API** — free footage API; attribution is added to the YouTube description.
4. **YouTube Data API** — official upload/metadata/thumbnail API.
5. **FFmpeg + Pillow** — open-source rendering and thumbnail creation.

## Setup

### 1. Create a GitHub repository

Recommended for a strict €0 build: a **public** repository so standard GitHub-hosted Actions runners are free. Secrets remain stored in GitHub Actions Secrets; do not commit them.

Upload this project to the repository.

### 2. Create a free Gemini API key

Create a key in Google AI Studio and save it in GitHub:

`Settings → Secrets and variables → Actions → New repository secret`

Secret name:

`GEMINI_API_KEY`

### 3. Create a free Pexels API key

Create a Pexels API key and save it as:

`PEXELS_API_KEY`

Pexels attribution is automatically appended to each video description.

### 4. Create YouTube OAuth credentials

In Google Cloud:

- create/select a project;
- enable **YouTube Data API v3**;
- configure OAuth consent;
- create an OAuth Desktop client;
- download the client secrets JSON.

You need these GitHub secrets:

- `YOUTUBE_CLIENT_ID`
- `YOUTUBE_CLIENT_SECRET`
- `YOUTUBE_REFRESH_TOKEN`

No local server is required for the agent. For the one-time OAuth authorization you can use Google OAuth 2.0 Playground in the browser with your own OAuth client, or optionally use the included helper `scripts/get_youtube_refresh_token.py` in any trusted environment. Keep all credentials private.

### 5. Add GitHub Actions secrets

Add all five:

- `GEMINI_API_KEY`
- `PEXELS_API_KEY`
- `YOUTUBE_CLIENT_ID`
- `YOUTUBE_CLIENT_SECRET`
- `YOUTUBE_REFRESH_TOKEN`

### 6. Test manually first

Open `Actions → Generate and publish → Run workflow` and choose `short`.

The workflow generates and uploads a private YouTube video. Inspect the first few outputs manually before relying on the schedule.

## Default schedule

The workflow currently has six daily slots:

- 09:30 UTC — Short
- 12:30 UTC — Short
- 15:30 UTC — Short
- 18:00 UTC — Long
- 20:00 UTC — Short
- 22:00 UTC — optional fifth Short

GitHub cron is UTC. Europe/Rome shifts between UTC+1 and UTC+2, so the local publishing time moves one hour when daylight-saving time changes. A timezone-aware scheduler can be added later.

## TikTok

V1 exports a 9:16 Short that is technically suitable for TikTok, but it does not bypass TikTok's posting requirements. The next TikTok step should use either an approved publishing partner or an officially permitted TikTok Content Posting integration with the required user-control/consent flow.

## Rights ledger

Each production run creates `rights_ledger.json`, recording:

- script source;
- TTS model;
- Pexels video IDs/creators/URLs;
- whether synthetic-media disclosure is set;
- zero-budget safety state.

## Important production note

A zero-budget automated channel should still be reviewed during the initial calibration period. The system is designed to reject paid services, not to guarantee that every generated story or stock-footage selection will be editorially perfect.
