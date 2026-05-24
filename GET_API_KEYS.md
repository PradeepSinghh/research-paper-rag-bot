# How to Get Your API Keys

This guide walks you through getting your **Groq API key** and **Cohere API key** — both are free to start.

---

## 1. Groq API Key

Groq provides blazing-fast LLM inference (used for answer generation).

### Steps

1. **Go to the Groq Console**
   → [https://console.groq.com](https://console.groq.com)

2. **Sign up / Log in**
   - Click **Sign Up** if you don't have an account.
   - You can sign up with Google, GitHub, or email.

3. **Verify your email** (if using email sign-up)
   - Check your inbox and click the verification link.

4. **Go to API Keys**
   - After logging in, click your profile icon (top-right).
   - Select **API Keys** from the dropdown.
   - Or go directly to: [https://console.groq.com/keys](https://console.groq.com/keys)

5. **Create a new key**
   - Click **Create API Key**.
   - Give it a name (e.g., `research-rag-bot`).
   - Click **Submit**.

6. **Copy your key immediately**
   - The key is shown **only once**. Copy it now.
   - It starts with `gsk_...`

7. **Paste it into your `.env` file**
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

### Free tier limits (as of 2026)
| Model | Requests/min | Tokens/min | Tokens/day |
|---|---|---|---|
| llama-3.3-70b-versatile | 30 | 6,000 | 100,000 |

> Enough for dozens of research paper sessions per day at no cost.

---

## 2. Cohere API Key

Cohere provides embeddings (to index your papers) and reranking (to improve retrieval quality).

### Steps

1. **Go to the Cohere Dashboard**
   → [https://dashboard.cohere.com](https://dashboard.cohere.com)

2. **Sign up / Log in**
   - Click **Start for free**.
   - Sign up with Google, GitHub, or email.

3. **Verify your email** (if using email sign-up)
   - Check your inbox and confirm your account.

4. **Go to API Keys**
   - In the left sidebar, click **API Keys**.
   - Or go directly to: [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)

5. **Copy the default key** (or create a new one)
   - A default key is already created for you.
   - Click **Copy** next to it.
   - Or click **New Trial Key** to create a named key.

6. **Paste it into your `.env` file**
   ```env
   COHERE_API_KEY=your_key_here
   ```

### Free tier limits (as of 2026)
| Feature | Free (Trial key) |
|---|---|
| Embed requests | 1,000 / month |
| Rerank requests | 1,000 / month |
| Rate limit | 100 requests / minute |

> Sufficient for indexing and querying several research papers during development.  
> Upgrade to a Production key for higher limits (still has a free tier).

---

## 3. Add Keys to Your Project

Once you have both keys:

```bash
# Open the .env file in the project root
nano "/Users/pradeepsingh/AI Bot/research_paper_bot/research-paper-rag-chatbot/.env"
```

Replace the placeholder values:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Save and close (`Ctrl+O`, `Enter`, `Ctrl+X` in nano).

---

## 4. Restart the App

The app reads `.env` at startup. If it's already running, **stop and restart** it:

```bash
# Stop: press Ctrl+C in the terminal running the app

# Restart:
cd "research-paper-rag-chatbot"
source .venv/bin/activate
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — the API key warning will be gone and you're ready to upload papers.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is not set` after adding key | Make sure the file is saved as `.env` (not `.env.txt`) and restart the app |
| `401 Unauthorized` from Cohere | Key was copied with extra whitespace — re-paste it carefully |
| Cohere embed returns 429 | Free tier rate limit hit — wait 60 seconds and retry, or upload one paper at a time |
| Groq returns 429 | You hit the free RPM limit — wait a moment and ask again |
| Key starts with `sk-` instead of `gsk_` | That's an OpenAI key — make sure you're on [console.groq.com](https://console.groq.com) |
