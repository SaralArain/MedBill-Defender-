"""
MedBill Defender
An AI agent that reads a medical bill, finds billing errors,
and drafts an insurance appeal / dispute letter.

Stack: Streamlit + Google Gemini (free tier) + PyMuPDF + fpdf2
"""

import io
import json
import re
from datetime import datetime

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from fpdf import FPDF
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MedBill Defender", page_icon="🛡️", layout="wide")

MODELS = [MODELS = ["gemini-3.6-flash"]
MAX_IMAGES = 8
MAX_TEXT_CHARS = 30000

# ----------------------------------------------------------------------------
# PROMPTS
# ----------------------------------------------------------------------------
EXTRACT_PROMPT = """You are a medical billing data extractor.

Read the attached medical bill (image and/or text) and return STRICT JSON ONLY.

Schema:
{
  "provider_name": string|null,
  "provider_address": string|null,
  "patient_name": string|null,
  "patient_account_number": string|null,
  "date_of_service": string|null,
  "bill_date": string|null,
  "insurance_name": string|null,
  "policy_number": string|null,
  "total_billed": number|null,
  "insurance_paid": number|null,
  "adjustments": number|null,
  "patient_responsibility": number|null,
  "line_items": [
    {
      "cpt_code": string|null,
      "description": string,
      "date": string|null,
      "quantity": number|null,
      "unit_charge": number|null,
      "total_charge": number|null
    }
  ],
  "notes": string|null
}

Rules:
- Numbers must be plain (no $, no commas).
- If a field is missing or unreadable, use null. Never invent data.
- Include EVERY line item you can read, even small ones.
- Return only the JSON object. No markdown, no commentary.
"""

ANALYZE_PROMPT = """You are a certified medical billing advocate with 20 years of US experience.

You will receive structured JSON extracted from a patient's medical bill.
Find every plausible billing error or overcharge and estimate realistic savings.

Check specifically for:
1. DUPLICATE CHARGES - same code, same date, same amount billed twice.
2. UPCODING - a higher-level (pricier) code than the service described.
3. UNBUNDLING - services that should be one bundled code split into many.
4. BALANCE BILLING - patient billed beyond their plan's allowed amount/copay.
5. SERVICES NOT RENDERED - items that look like they never happened.
6. MATH ERRORS - line items that do not sum to the stated total.
7. UNCLEAR CHARGES - vague descriptions, "misc", missing CPT codes.
8. EXCESSIVE CHARGES - unit prices far above typical Medicare allowable rates.
9. FACILITY FEES - a separate "facility" charge on a simple office visit.
10. MISSING ADJUSTMENT - insurance payment/adjustment not applied.

Be honest. Do not fabricate issues. If the bill looks clean, say so and set bill_clean true.

Return STRICT JSON ONLY with this exact schema:
{
  "summary": "2-4 sentence plain-English summary",
  "bill_clean": true,
  "estimated_savings_low": 0,
  "estimated_savings_high": 0,
  "confidence": "low",
  "issues": [
    {
      "id": 1,
      "type": "Duplicate charge",
      "severity": "high",
      "line_reference": "line 3 and line 7",
      "amount_at_stake": 0,
      "explanation": "plain English why this is wrong",
      "recommended_action": "exactly what the patient should do"
    }
  ],
  "questions_to_ask": ["..."],
  "next_steps": ["..."],
  "disclaimer": "short note that this is not legal or medical advice"
}
severity must be one of: high, medium, low.
confidence must be one of: low, medium, high.
"""

LETTER_PROMPT = """You are a professional patient advocate. Write a formal, firm, polite dispute letter.

CONTEXT
- Patient name: {patient_name}
- Patient address: {patient_address}
- Insurance member ID: {member_id}
- Provider / billing entity: {provider}
- Account number: {account}
- Date(s) of service: {dos}
- Amount in dispute: {amount}
- Desired outcome: {outcome}

ISSUES FOUND
{issues_text}

INSTRUCTIONS
- Write a complete letter ready to print or email. Today's date: {today}.
- Address it to the provider's billing department.
- Open with account details so they can identify the claim.
- Explain each disputed item clearly, referencing CPT codes and dollar amounts.
- Cite patient rights: right to an itemized bill, the No Surprises Act where applicable,
  and the right to appeal.
- Request a corrected bill and a written response within 30 days.
- Ask them to place the account on hold and pause any collections activity during the dispute.
- Professional, non-accusatory tone. No threats.
- End with a signature block for the patient.
- Output ONLY the letter text. No markdown fences, no commentary.
"""

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "—"


def parse_json(text: str) -> dict:
    """Robust JSON parse — strips fences, falls back to first {...} block."""
    if not text:
        raise ValueError("Empty response from model.")
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def get_api_key() -> str:
    """Priority: Streamlit secrets > sidebar input."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return st.session_state.get("api_key_input", "").strip()


def load_document(uploaded_files):
    """Return (text, [PIL images]). Uses PDF text layer when available."""
    text_parts, images = [], []
    for f in uploaded_files:
        data = f.getvalue()
        name = f.name.lower()
        if name.endswith(".pdf"):
            doc = fitz.open(stream=data, filetype="pdf")
            pdf_text = "\n".join(page.get_text() for page in doc)
            if len(pdf_text.strip()) > 120:
                text_parts.append(pdf_text)
            else:
                for page in doc:
                    if len(images) >= MAX_IMAGES:
                        break
                    pix = page.get_pixmap(dpi=150)
                    images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
            doc.close()
        else:
            if len(images) < MAX_IMAGES:
                images.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return "\n\n".join(text_parts)[:MAX_TEXT_CHARS], images


def build_parts(prompt: str, text: str, images: list):
    parts = [prompt]
    if text:
        parts.append("--- BILL TEXT ---\n" + text)
    parts.extend(images[:MAX_IMAGES])
    return parts


def call_gemini(api_key: str, model_name: str, parts: list, json_mode: bool = True) -> str:
    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    resp = client.models.generate_content(model=model_name, contents=parts, config=cfg)
    return (resp.text or "").strip()


def text_to_pdf(text: str) -> bytes:
    """Render plain text into a simple PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.split("\n"):
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
    return bytes(pdf.output())


def issues_to_text(analysis: dict) -> str:
    issues = analysis.get("issues") or []
    if not issues:
        return "No specific billing errors were identified."
    out = []
    for i in issues:
        out.append(
            f"- [{str(i.get('severity','')).upper()}] {i.get('type','Issue')} "
            f"({i.get('line_reference','bill')}) — {money(i.get('amount_at_stake'))} at stake.\n"
            f"  Why: {i.get('explanation','')}\n"
            f"  Action: {i.get('recommended_action','')}"
        )
    return "\n".join(out)


def init_state():
    defaults = {
        "bill": None,
        "analysis": None,
        "letter": "",
        "disputes": [],
        "api_key_input": "",
        "model_name": MODELS[0],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
init_state()

st.title("🛡️ MedBill Defender")
st.caption("Upload a medical bill → the agent finds billing errors → drafts your appeal letter.")

# ---- sidebar ----
with st.sidebar:
    st.header("⚙️ Setup")

    key_ok = False
    try:
        key_ok = "GEMINI_API_KEY" in st.secrets and bool(st.secrets["GEMINI_API_KEY"])
    except Exception:
        key_ok = False

    if key_ok:
        st.success("API key loaded from secrets ✅")
    else:
        st.warning("No key in secrets. Paste one below (local dev only).")
        st.text_input(
            "Gemini API key",
            type="password",
            key="api_key_input",
            help="Get a free key at https://aistudio.google.com/apikey",
        )

    st.session_state["model_name"] = st.selectbox(
        "Model", MODELS, index=MODELS.index(st.session_state["model_name"])
    )

    st.divider()
    st.markdown(
        "**Free stack**\n"
        "- Streamlit Community Cloud\n"
        "- Google Gemini free tier\n"
        "- GitHub for hosting\n\n"
        "⚠️ This tool is not legal or medical advice."
    )

API_KEY = get_api_key()

# ---- tabs ----
tab_upload, tab_analysis, tab_letter, tab_track = st.tabs(
    ["📤 1. Upload & Read", "🔍 2. Find Errors", "✉️ 3. Appeal Letter", "📋 4. My Disputes"]
)

# ============================ TAB 1 ============================
with tab_upload:
    st.subheader("Upload your bill")
    files = st.file_uploader(
        "Photos or PDF of your medical bill / EOB",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        read_btn = st.button("🔎 Read bill", type="primary", use_container_width=True)

    if read_btn:
        if not API_KEY:
            st.error("Add your Gemini API key first (sidebar).")
        elif not files:
            st.error("Upload at least one file.")
        else:
            with st.spinner("Reading the bill…"):
                try:
                    text, images = load_document(files)
                    if not text and not images:
                        st.error("Could not read that file.")
                    else:
                        raw = call_gemini(
                            API_KEY,
                            st.session_state["model_name"],
                            build_parts(EXTRACT_PROMPT, text, images),
                        )
                        st.session_state["bill"] = parse_json(raw)
                        st.session_state["analysis"] = None
                        st.session_state["letter"] = ""
                        st.success("Bill parsed ✅")
                except Exception as e:
                    st.error(f"Failed: {e}")

    bill = st.session_state.get("bill")
    if bill:
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Provider", (bill.get("provider_name") or "—")[:28])
        c2.metric("Total billed", money(bill.get("total_billed")))
        c3.metric("Insurance paid", money(bill.get("insurance_paid")))
        c4.metric("You owe", money(bill.get("patient_responsibility")))

        with st.expander("📄 Extracted details", expanded=False):
            st.json(bill)

        items = bill.get("line_items") or []
        if items:
            st.markdown("**Line items**")
            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

# ============================ TAB 2 ============================
with tab_analysis:
    st.subheader("Error detection")
    bill = st.session_state.get("bill")

    if not bill:
        st.info("Go to **Upload & Read** first.")
    else:
        if st.button("🧠 Analyze for errors", type="primary"):
            if not API_KEY:
                st.error("Add your Gemini API key first.")
            else:
                with st.spinner("Auditing your bill…"):
                    try:
                        parts = [ANALYZE_PROMPT, json.dumps(bill, indent=2)]
                        raw = call_gemini(API_KEY, st.session_state["model_name"], parts)
                        st.session_state["analysis"] = parse_json(raw)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    analysis = st.session_state.get("analysis")
    if analysis:
        st.divider()
        st.markdown("### Summary")
        st.write(analysis.get("summary", ""))

        m1, m2, m3 = st.columns(3)
        lo = analysis.get("estimated_savings_low") or 0
        hi = analysis.get("estimated_savings_high") or 0
        m1.metric("Est. recoverable", f"{money(lo)} – {money(hi)}")
        m2.metric("Confidence", str(analysis.get("confidence", "—")).title())
        m3.metric("Issues found", len(analysis.get("issues") or []))

        if analysis.get("bill_clean"):
            st.success("No clear errors detected. Still worth requesting an itemized bill.")

        for issue in analysis.get("issues") or []:
            sev = str(issue.get("severity", "low")).lower()
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(sev, "⚪")
            title = (
                f"{icon} {issue.get('type','Issue')} — "
                f"{money(issue.get('amount_at_stake'))} · {issue.get('line_reference','')}"
            )
            with st.expander(title):
                st.markdown(f"**Why it's wrong:** {issue.get('explanation','')}")
                st.markdown(f"**What to do:** {issue.get('recommended_action','')}")

        q = analysis.get("questions_to_ask") or []
        if q:
            st.markdown("### ❓ Ask the billing office")
            for item in q:
                st.markdown(f"- {item}")

        n = analysis.get("next_steps") or []
        if n:
            st.markdown("### ➡️ Next steps")
            for item in n:
                st.markdown(f"- {item}")

        st.caption(analysis.get("disclaimer", "Not legal or medical advice."))

# ============================ TAB 3 ============================
with tab_letter:
    st.subheader("Generate your appeal letter")
    bill = st.session_state.get("bill")
    analysis = st.session_state.get("analysis")

    if not bill or not analysis:
        st.info("Run **Upload & Read** and **Find Errors** first.")
    else:
        with st.form("letter_form"):
            c1, c2 = st.columns(2)
            patient_name = c1.text_input("Your name", value=bill.get("patient_name") or "")
            member_id = c2.text_input("Insurance member ID", value=bill.get("policy_number") or "")
            patient_address = c1.text_input("Your address", value="")
            account = c2.text_input(
                "Account / claim number", value=bill.get("patient_account_number") or ""
            )
            provider = c1.text_input("Provider / billing dept", value=bill.get("provider_name") or "")
            dos = c2.text_input("Date(s) of service", value=bill.get("date_of_service") or "")
            amount = c1.text_input(
                "Amount in dispute",
                value=str(bill.get("patient_responsibility") or bill.get("total_billed") or ""),
            )
            outcome = c2.selectbox(
                "Desired outcome",
                [
                    "Remove incorrect charges and send a corrected bill",
                    "Reprocess the claim with my insurance",
                    "Reduce the balance to the correct patient responsibility",
                    "Full review with an itemized statement",
                ],
            )
            submitted = st.form_submit_button("✍️ Draft letter", type="primary")

        if submitted:
            if not API_KEY:
                st.error("Add your Gemini API key first.")
            else:
                prompt = LETTER_PROMPT.format(
                    patient_name=patient_name or "[Patient Name]",
                    patient_address=patient_address or "[Address]",
                    member_id=member_id or "[Member ID]",
                    provider=provider or "[Provider]",
                    account=account or "[Account #]",
                    dos=dos or "[Date of Service]",
                    amount=amount or "[Amount]",
                    outcome=outcome,
                    issues_text=issues_to_text(analysis),
                    today=datetime.now().strftime("%B %d, %Y"),
                )
                with st.spinner("Writing your letter…"):
                    try:
                        st.session_state["letter"] = call_gemini(
                            API_KEY, st.session_state["model_name"], [prompt], json_mode=False
                        )
                    except Exception as e:
                        st.error(f"Letter failed: {e}")

        if st.session_state.get("letter"):
            st.divider()
            letter = st.text_area("Edit before sending", st.session_state["letter"], height=460)

            d1, d2, d3 = st.columns(3)
            d1.download_button(
                "⬇️ Download .txt", letter, file_name="appeal_letter.txt", use_container_width=True
            )
            d2.download_button(
                "⬇️ Download .pdf",
                text_to_pdf(letter),
                file_name="appeal_letter.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            if d3.button("💾 Save to My Disputes", use_container_width=True):
                st.session_state["disputes"].append(
                    {
                        "saved": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "provider": provider or "—",
                        "account": account or "—",
                        "amount": amount or "—",
                        "issues": len(analysis.get("issues") or []),
                        "status": "Draft ready",
                    }
                )
                st.success("Saved.")

# ============================ TAB 4 ============================
with tab_track:
    st.subheader("Dispute tracker")
    disputes = st.session_state.get("disputes") or []
    if not disputes:
        st.info("No disputes saved yet.")
    else:
        df = pd.DataFrame(disputes)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="medbill_disputes.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Clear all"):
            st.session_state["disputes"] = []
            st.rerun()

st.divider()
st.caption("MedBill Defender · Educational tool. Not legal, medical, or financial advice.")
