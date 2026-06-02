# GitMind

A RAG-powered chatbot that indexes any Git repository into Pinecone and lets you ask questions about the codebase via a Streamlit web UI. Embeddings use Gemini, generation uses Gemini chat.

---

## Running Locally

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com) API key
- A [Pinecone](https://www.pinecone.io) account with an index (or let the app create one)

### 1. Clone and install

```bash
git clone https://github.com/NicoAbrs/gitmind.git
cd gitmind
pip install -r requirements.txt
```

### 2. Create a `.env` file

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

> The `.env` file is gitignored and never committed.

### 3. Run the app

```bash
streamlit run app/ui.py
```

The app opens at `http://localhost:8501`. Paste a public GitHub repository URL in the sidebar and click **Load Repository** to index it, then start chatting.

---

## Deploying to Streamlit Cloud

### 1. Push to GitHub

Make sure your code is pushed to a GitHub repository. The `.env` file should **not** be committed — secrets are added separately in the next step.

### 2. Create the app on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and set the main file path to `app/ui.py`.
4. Click **Advanced settings** and add the following secrets:

```toml
GOOGLE_API_KEY = "your_google_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX_NAME = "your_index_name"
```

5. Click **Deploy**.

Streamlit Cloud will install dependencies from `requirements.txt` automatically. Once deployed, share the generated URL with anyone.

### Notes

- Indexed vectors are stored in Pinecone, so they persist across app restarts and re-deploys.
- The BM25 sparse encoder (`bm25.json`) is rebuilt locally on each load from the cloned repo — it is not persisted.
- Streamlit Cloud has ephemeral storage, so the cloned repo (`./cloned_repo`) is temporary and re-cloned on each session.
