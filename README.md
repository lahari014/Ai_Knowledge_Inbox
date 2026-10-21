# AI Knowledge Inbox

A small web app for one person. Save a note or a public web page, then ask a question. The answer uses only what you saved, and the screen shows the exact passages it used.

This is a retrieval-augmented generation app, usually called RAG:

1. Save the original note or page.
2. Split it into short passages.
3. Turn each passage into a list of numbers (an embedding) so similar text can be found.
4. Store those lists in a vector database.
5. Turn the question into the same kind of list and keep the closest passages.
6. Answer from those passages and return them as citations.

```text
Browser
  |  POST /api/ingest     GET /api/items     POST /api/query
  v
Vite (local) or Nginx (Docker) strips the /api prefix
  |
  v
FastAPI
  |-- SQLite     original notes and pages
  |-- Chroma     passages and nearest-neighbor search
  |-- OpenAI     embeddings and written answers, only when a key is set
  `-- demo mode  local embeddings and quoted passages, when no key is set
```

The browser never calls OpenAI, Chroma, or the page being saved. There are no accounts, no editing, and no deletion.

## What a reviewer can check

| Assignment requirement | Where it is |
| --- | --- |
| Save a plain-text note | Add knowledge → Note |
| Save a URL and fetch it on the server | Add knowledge → URL |
| Store the raw text, source type, and time | SQLite, listed under Saved knowledge |
| Chunk, embed, and search | `backend/app/services/chunking.py`, `embeddings.py`, Chroma |
| Answer from the retrieved passages and cite them | Ask your knowledge |
| `POST /ingest`, `GET /items`, `POST /query` | `backend/app/api/routes.py` |
| Validation and error codes | One error shape, status 422, 502, 503, or 504 |
| React with hooks | `frontend/src` |

## Requirements

- Docker, for the one-command path
- Or, for the local path: Python 3.10 or newer, Node.js 20 or newer, and pip

Leave `OPENAI_API_KEY` empty. The app then runs in demo mode and does not need an OpenAI account.

## Run it with Docker

From the repository root:

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080).

API docs are at [http://localhost:8080/api/docs](http://localhost:8080/api/docs).

Stop it with:

```bash
docker compose down
```

Saved data stays in Docker volumes. `docker compose down -v` deletes that data too.

## Run it locally

Use three terminals. Chroma uses port 8001 and the API uses port 8000. Chroma 0.5 needs an older FastAPI than this API, so Chroma gets its own virtual environment.

Terminal 1, from the repository root:

```bash
python -m venv .chroma-venv
.chroma-venv/bin/python -m pip install 'chromadb==0.5.23' 'fastapi<0.116'
.chroma-venv/bin/chroma run --path ./backend/data/chroma --port 8001
```

Terminal 2:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp ../.env.example .env
uvicorn app.main:create_app --factory --reload --port 8000
```

Terminal 3:

```bash
cd frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The page calls `/api/...`. Vite removes `/api` and forwards the request to `http://localhost:8000`.

If `python -m venv` says `ensurepip` is missing, install [uv](https://docs.astral.sh/uv/) and create the backend environment with `uv venv --python 3.10 .venv`, then install the requirements with `uv pip install --python .venv/bin/python -r requirements-dev.txt`.

## Try it

1. Leave the Note tab selected. Enter a title and this text: `The launch date is 12 May.`
2. Click **Save note**. The card appears under Saved knowledge.
3. Ask `When is the launch?`
4. The answer includes `12 May`. Under it, the source card shows the same passage and a similarity score.
5. Optional: open the URL tab, save `https://example.com`, and ask what the page is about.

A passage is used only when its similarity is at least 0.15 and at least half the score of the closest passage. Weaker passages are left out of both the answer and the source list. If nothing qualifies, the answer is: `I don't have enough saved knowledge to answer that.`

## Demo mode and OpenAI

Demo mode is the default. Embeddings are a stable local hash of words and phrases. The answer quotes the retrieved passages and is labeled **Demo mode**. This is enough to review the whole pipeline without an API key. It is not as good as a model at matching a paraphrase.

To use OpenAI, edit `backend/.env`:

```bash
OPENAI_API_KEY=sk-your-personal-key
CHROMA_COLLECTION=knowledge-openai
```

The models are `text-embedding-3-small` and `gpt-4o-mini`. Restart the API after changing `.env`. OpenAI vectors have 1536 numbers and demo vectors have 384, so they cannot share one Chroma collection. Startup stops if the saved collection was built by the other provider. Switch `CHROMA_COLLECTION` back to `knowledge` when you remove the key.

The key stays in `backend/.env`. That file is gitignored. Do not put a key in `.env.example`.

## API

Errors all look like this. The same id is in the `x-request-id` response header.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Question cannot be empty.",
    "requestId": "req-123"
  }
}
```

Save a note:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"type":"note","title":"Launch","text":"The launch date is 12 May."}'
```

Save a public page:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"type":"url","url":"https://example.com"}'
```

A successful save returns `201` and a short preview of at most 180 characters. A note has no `url` field. Invalid input returns `422` and stores nothing.

List what was saved, newest first:

```bash
curl -s http://localhost:8000/items
```

Ask a question. `limit` is optional, from 1 to 10, and defaults to 5.

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"When is the launch?","limit":5}'
```

The response has `answer`, `mode` (`demo` or `openai`), and `citations`. Each citation has a number, item id, label, the exact snippet, a score from 0 to 1, and a URL when the source is a page.

Health check:

```bash
curl -s http://localhost:8000/health
```

Interactive docs for the local API are at [http://localhost:8000/docs](http://localhost:8000/docs).

## Tests

These tests do not call OpenAI or the network. The backend uses an in-memory vector store and a fake page fetcher.

Backend:

```bash
cd backend
source .venv/bin/activate
ruff check .
ruff format --check .
mypy app
pytest --cov=app -q
```

Frontend:

```bash
cd frontend
npm run lint
npm run typecheck
npm test -- --run
npm run build
```

Check the Compose file without starting it:

```bash
docker compose config
```

## Design choices

**Chunking.** A passage is about 800 characters, and at most 120 characters are repeated from the previous passage. The splitter prefers a paragraph break, then a sentence, and only then a hard cut. The overlap keeps a sentence that sits on a boundary visible to a question about its second half.

**SQLite and Chroma.** SQLite stores the original item: id, note or URL, title, raw text, and the UTC time. Chroma answers "which passages are nearest to this question?" At this size, scanning vectors inside SQLite would also work. Chroma makes that search a separate step, which is what a larger system would do.

**Saving waits until indexing finishes.** The request does not return until the page is fetched, saved, and embedded. That is right for a short note. It is the wrong shape for a large crawl.

**A failed save leaves nothing behind.** SQLite and Chroma cannot share one transaction. If embedding fails after the row is inserted, the API deletes any passages for that item and then deletes the row.

**URL fetching is restricted.** The server fetches the page, so a user-supplied URL could be aimed at an internal address. Only HTTP and HTTPS are allowed. Credentials in the URL are rejected. The hostname is resolved, and loopback, private, link-local, and reserved addresses are blocked. Redirects are checked again, at most 3 times. The response is limited to 2 MB and must be HTML or plain text. Script, style, navigation, and footer tags are removed before the text is stored.

**Saved text is data.** In OpenAI mode the model is told to answer only from the numbered passages and to ignore instructions found inside them. Citations are returned as fields, not scraped out of the sentence.

**Logs.** Each request writes one JSON line with the request id, method, path, status, duration, and stage. Logs do not include the API key, the saved text, or the model prompt.

## What breaks as this grows

- The HTTP request times out when embedding or fetching takes too long. A queue with job status is the replacement.
- One SQLite file and one Chroma collection do not isolate users and do not take multi-server writes safely. PostgreSQL and a managed vector service would.
- Changing the embedding model makes old vectors unusable. Reindexing has to be a separate step. That is why demo mode and OpenAI mode use different collection names.
- A fetched page can still contain hostile text. The prompt treats it as untrusted. A production fetcher would also run in a locked-down network.

A production version would add authentication, rate limits, tracing, cost limits, and backups. Those are left out here on purpose. The assignment asks for the pipeline, not an account system.
