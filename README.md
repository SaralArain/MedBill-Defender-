# 🛡️ MedBill Defender

An AI agent that reads a medical bill, finds billing errors, and drafts an
insurance appeal letter — built with Streamlit + Google Gemini.

## Features
- 📤 Upload photos or PDF of a medical bill / EOB
- 🔎 Gemini extracts provider, totals, and every line item
- 🧠 Detects duplicate charges, upcoding, unbundling, balance billing,
  math errors, and excessive charges
- ✉️ Drafts a formal dispute letter you can edit, download as TXT/PDF
- 📋 Tracks your disputes in-session and exports CSV

## Run locally
```bash
git clone https://github.com/<you>/medbill-defender.git
cd medbill-defender
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml and paste your real Gemini key

streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Push this repo to GitHub (make sure `secrets.toml` is gitignored).
2. Go to https://share.streamlit.io → **New app** → pick your repo → `app.py`.
3. Click **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
4. Deploy. Done — free HTTPS URL.

## Notes
- Gemini free tier has rate limits; if you hit one, wait a minute or switch models
  in the sidebar.
- If a model name errors, change `MODELS` at the top of `app.py`.
- Educational tool. Not legal, medical, or financial advice.
