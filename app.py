"""
Hospital Readmission Risk + Diabetes Guideline RAG Assistant
================================================================
Single-page layout: optional patient intake form at top, followed by
one unified Q&A section below (plain inline widgets, not floating).
If patient data has been submitted, answers are personalized to that
patient's risk profile. Otherwise, questions are answered generally
from the guidelines alone.

Deployment target: Streamlit Community Cloud

Required files in the same repo as this app.py:
- xgb_readmission_model.pkl
- model_columns.pkl
- chroma_db/               (the persisted Chroma vector database folder)
- requirements.txt
- .streamlit/config.toml
"""

import re
import random
import pandas as pd
import streamlit as st
import joblib
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ==========================================================================
# PAGE CONFIG
# ==========================================================================
st.set_page_config(page_title="Diabetes Readmission Risk Assistant", layout="wide")

# ==========================================================================
# CUSTOM CSS — matches the analyzewithrutuja.github.io portfolio theme
# ==========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

:root {
    --navy: #190d26;
    --pink: #e8187a;
    --pink2: #f9004d;
    --muted: #7a7a8c;
    --border: #e0e0e0;
    --shadow: 0 4px 24px rgba(25,13,38,0.08);
    --shadow-lg: 0 12px 40px rgba(232,24,122,0.16);
}

.stApp { background-color: #f0f0f0; }

h1, h2, h3 { color: var(--navy) !important; font-weight: 800 !important; }

/* ---------- Header ---------- */
.app-eyebrow {
    font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}
.app-eyebrow::before { content: ''; width: 24px; height: 1px; background: var(--muted); }

.app-title { font-size: 2.4rem; font-weight: 800; color: var(--navy); margin-bottom: 6px; letter-spacing: -0.01em; }
.app-title .pink {
    background: linear-gradient(135deg, #e8187a, #f9004d);
    -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 900;
}

.app-sub { font-size: 0.87rem; color: var(--muted); margin-bottom: 32px; max-width: 720px; line-height: 1.75; }

/* ---------- Cards ---------- */
div[data-testid="stForm"] {
    background: #ffffff;
    border: 1.5px solid var(--border);
    border-radius: 18px;
    padding: 26px 30px;
    box-shadow: var(--shadow);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1.5px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: var(--shadow);
    overflow: hidden;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
    margin-bottom: 20px;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--shadow-lg);
}

.card-accent {
    height: 4px;
    background: linear-gradient(90deg, #e8187a, #f9004d);
    margin: -1px -1px 20px -1px;
}

.section-label {
    font-size: 1rem; font-weight: 700; color: var(--navy);
    margin: 20px 0 12px 0; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 9px;
    letter-spacing: -0.01em;
}
.section-label::before {
    content: ''; width: 7px; height: 7px; border-radius: 50%;
    background: linear-gradient(135deg, #e8187a, #f9004d); flex-shrink: 0;
}

.helper-text {
    font-size: 0.78rem; color: var(--muted); font-style: italic; margin-bottom: 16px;
    padding: 8px 12px; background: rgba(232,24,122,0.04); border-radius: 8px;
    border-left: 2px solid rgba(232,24,122,0.3);
}

/* ---------- Buttons ---------- */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #e8187a, #f9004d);
    color: white; border: none; border-radius: 10px;
    font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    font-size: 0.78rem; padding: 11px 26px;
    transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    opacity: 0.94; transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(232,24,122,0.4);
}
.stButton > button:active, .stFormSubmitButton > button:active {
    transform: translateY(0); opacity: 1;
}

/* ---------- Risk display ---------- */
.risk-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: white; margin-top: 8px;
}
.risk-high { background: linear-gradient(135deg, #e8187a, #f9004d); }
.risk-moderate { background: #e8187a; }
.risk-low { background: var(--navy); opacity: 0.8; }

.risk-eyebrow {
    font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 600; margin-bottom: 2px;
}
.risk-score-number { font-size: 2.6rem; font-weight: 800; color: var(--navy); line-height: 1.05; letter-spacing: -0.02em; }

/* Flag list */
.flag-item {
    font-size: 0.81rem; color: var(--navy);
    background: rgba(232,24,122,0.055);
    border-left: 3px solid var(--pink);
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 8px;
    transition: background 0.2s ease;
}
.flag-item:hover { background: rgba(232,24,122,0.09); }

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: #ffffff; border: 1.5px solid var(--border);
    border-radius: 18px; overflow: hidden; box-shadow: var(--shadow);
    margin-bottom: 20px;
    transition: box-shadow 0.3s ease;
}
[data-testid="stExpander"]:hover { box-shadow: var(--shadow-lg); }
[data-testid="stExpander"] summary {
    padding: 18px 26px; font-weight: 700; color: var(--navy);
    font-size: 0.94rem; background: #ffffff; transition: all 0.2s ease;
}
[data-testid="stExpander"] summary:hover {
    background: rgba(232, 24, 122, 0.045); color: var(--pink);
}
[data-testid="stExpander"] summary svg { fill: var(--pink) !important; transition: transform 0.2s ease; }
[data-testid="stExpander"] details[open] summary { border-bottom: 1.5px solid var(--border); }

/* ---------- Q&A entries ---------- */
.qa-question {
    font-weight: 700; color: var(--navy); margin-top: 18px;
    padding-left: 14px; border-left: 3px solid var(--pink); font-size: 0.89rem;
}
.qa-answer {
    font-size: 0.86rem; color: var(--navy); line-height: 1.8;
    margin: 10px 0 18px 14px; padding: 16px 18px;
    background: #faf7f9; border-radius: 12px;
    border: 1px solid rgba(232,24,122,0.08);
}

/* ---------- Search-bar pill (targeted via st.container key="search_bar") ---------- */
.st-key-search_bar {
    position: relative !important;
    border-radius: 999px !important;
    padding: 4px 2px 4px 22px !important;
    box-shadow: none !important;
    border: 1.5px solid var(--border) !important;
    margin-bottom: 0 !important;
    background: #ffffff !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.st-key-search_bar:hover {
    border-color: #d8d8e0 !important;
}
.st-key-search_bar:focus-within {
    border-color: var(--pink) !important;
    box-shadow: 0 0 0 3px rgba(232,24,122,0.1) !important;
}

/* Remove Streamlit/BaseWeb's own input border + red focus ring so only
   our pill container's border shows — target every descendant to be safe */
.st-key-search_bar [data-testid="stTextInput"] * {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

.st-key-search_bar [data-testid="stTextInput"] input {
    background-color: transparent !important;
    padding: 10px 46px 10px 6px !important;
    height: 40px !important;
    font-size: 0.9rem !important;
    color: var(--navy) !important;
}
.st-key-search_bar [data-testid="stTextInput"] input:focus {
    box-shadow: none !important;
    outline: none !important;
}
.st-key-search_bar [data-testid="stTextInput"] input::placeholder { color: #a8a8b8 !important; }

[data-testid="stHorizontalBlock"] { align-items: center !important; }

/* Circular send button — pushed flush against the right edge, with the
   arrow color forced on both the button and its inner <p> tag so it is
   never invisible regardless of Streamlit's default text color */
/* Directly target the LAST column within our search-bar container (the
   one holding the send button) using descendant + :last-child — no :has()
   needed, so this works even in browsers that don't support :has() */
.st-key-search_bar [data-testid="column"]:last-child {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
}

/* Belt-and-suspenders: also PIN the button with absolute positioning,
   anchored to .st-key-search_bar (position:relative above). This does
   not depend on column width or flex space calculations at all — it
   simply places the button at a fixed pixel offset from the pill's
   right edge, vertically centered, regardless of layout quirks. */
html body .st-key-ask_button {
    position: absolute !important;
    right: 2px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    margin: 0 !important;
    z-index: 10 !important;
}
html body .st-key-ask_button button {
    background: linear-gradient(135deg, #e8187a, #f9004d) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    min-width: 40px !important;
    max-width: 40px !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    box-sizing: border-box !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 12px rgba(232,24,122,0.4) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}
html body .st-key-ask_button button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 6px 16px rgba(232,24,122,0.5) !important;
}
html body .st-key-ask_button button:active {
    transform: scale(0.95) !important;
}
html body .st-key-ask_button button p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--navy);
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(232,24,122,0.15) !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================================
# LOAD RESOURCES (cached so they only load once)
# ==========================================================================
@st.cache_resource
def load_model_resources():
    xgb_model = joblib.load("xgb_readmission_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return xgb_model, model_columns

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_db():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="diabetes_guidelines")
    return collection

xgb_model, model_columns = load_model_resources()
embed_model = load_embedding_model()
collection = load_vector_db()

# ==========================================================================
# REFERENCE VALUE LISTS
# ==========================================================================
AGE_BANDS = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']

ADMISSION_TYPE = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped"
}

DISCHARGE_DISPOSITION = {
    1: "Discharged to home", 2: "Transferred to another short term hospital",
    3: "Transferred to SNF", 4: "Transferred to ICF",
    5: "Transferred to another inpatient care institution",
    6: "Discharged to home with home health service",
    7: "Left against medical advice", 8: "Discharged to home under Home IV provider",
    11: "Expired", 13: "Hospice / home", 14: "Hospice / medical facility",
    22: "Transferred to rehab facility", 23: "Transferred to long term care hospital",
    24: "Transferred to nursing facility (Medicaid only)", 25: "Not Mapped"
}

ADMISSION_SOURCE = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available", 20: "Not Mapped"
}

MED_STATUS_OPTIONS = ["No", "Steady", "Up", "Down"]
DIAG_CATEGORIES = ["Diabetes", "Circulatory", "Respiratory", "Digestive", "Injury", "Other"]

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

# ==========================================================================
# PREPROCESSING
# ==========================================================================
def build_model_input(form_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([form_data.copy()])
    df_encoded = pd.get_dummies(df)
    df_encoded.columns = [re.sub(r'[\[\]<>]', '', col) for col in df_encoded.columns]
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_encoded

def get_risk_level(risk_score: float) -> str:
    if risk_score >= 0.6:
        return "High"
    elif risk_score >= 0.3:
        return "Moderate"
    return "Low"

def format_patient_context(ctx: dict) -> str:
    risk_level = get_risk_level(ctx["risk_score"])
    return f"""Patient Risk Profile:
- Readmission risk: {risk_level} ({ctx['risk_score']*100:.0f}%)
- Prior hospital admissions (past year): {ctx['prior_admissions']}
- Current medication count: {ctx['num_medications']}
- Discharge destination: {ctx['discharge_disposition']}
- A1C control: {ctx['a1c_level']}
- Time in hospital: {ctx['time_in_hospital']} day(s)"""

def get_risk_flags(ctx: dict) -> list:
    flags = []
    if ctx["risk_score"] >= 0.6:
        flags.append("HIGH READMISSION RISK — prioritize early follow-up (within 7 days)")
    if ctx["num_medications"] > 10:
        flags.append("POLYPHARMACY CONCERN — medication reconciliation is critical")
    if ctx["discharge_disposition"] == "Discharged to home" and ctx["prior_admissions"] > 2:
        flags.append("HOME DISCHARGE WITH HISTORY OF REPEAT ADMISSIONS — consider home health referral")
    if ctx["a1c_level"] == "High":
        flags.append("POOR GLYCEMIC CONTROL — reinforce diabetes self-management education")
    return flags

# ==========================================================================
# RAG PIPELINE
# ==========================================================================
def retrieve_chunks(query: str, top_k: int = 5) -> str:
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    chunks_text = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        chunks_text.append(f"[Source: {meta['source']}, {meta['filename']}]\n{doc}")
    return "\n\n---\n\n".join(chunks_text)

def build_prompt(query: str, patient_context) -> str:
    retrieved_context = retrieve_chunks(query, top_k=5)

    if patient_context:
        patient_summary = format_patient_context(patient_context)
        flags = get_risk_flags(patient_context)
        flags_text = "\n".join(f"- {f}" for f in flags) if flags else "None"
        patient_block = f"""PATIENT CONTEXT:
{patient_summary}

PRIORITY FLAGS:
{flags_text}"""
    else:
        patient_block = "PATIENT CONTEXT:\nNone provided — answer generally."

    return f"""You are a clinical decision-support assistant specialized in type 2
diabetes management. Answer the question below using ONLY the guideline
excerpts provided. If patient context is given, tailor your answer to it and
prioritize any flagged concerns. Cite the source organization (e.g., ADA,
WHO, CDC, Mayo) for each recommendation. If the guidelines don't address
something, say so rather than guessing.

{patient_block}

RELEVANT GUIDELINES:
{retrieved_context}

QUESTION:
{query}

Provide a clear, structured answer with source citations. Include a brief
disclaimer that this is a decision-support suggestion, not a replacement
for clinical judgment."""

def generate_answer(query: str, patient_context, hf_token: str) -> str:
    client_llm = InferenceClient(token=hf_token)
    prompt = build_prompt(query, patient_context)
    response = client_llm.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==========================================================================
# SIDEBAR
# ==========================================================================
st.sidebar.markdown("### Settings")
hf_token = st.sidebar.text_input("Hugging Face API token", type="password")

# ==========================================================================
# SESSION STATE
# ==========================================================================
if "patient_context" not in st.session_state:
    st.session_state.patient_context = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================================================
# HEADER
# ==========================================================================
st.markdown('<div class="app-eyebrow">Clinical decision support</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Diabetes Readmission <span class="pink">Risk Assistant</span></div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Enter a patient\'s details to get a readmission risk score, or skip straight to asking a question — the assistant answers from ADA, WHO, CDC, and Mayo Clinic guidelines either way.</div>', unsafe_allow_html=True)

# ==========================================================================
# PATIENT FORM (optional, expandable)
# ==========================================================================
with st.expander("Fill in patient details to see their readmission risk", expanded=False):
    with st.form("patient_form"):
        st.markdown(
            '<div class="helper-text">Encounter ID: '
            f'{random.randint(100000, 999999)} (auto-generated, display only). '
            'Adjust the values below to match your patient, then click Assess Risk.</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">Demographics & Admission</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age = c1.selectbox("Age band", AGE_BANDS, index=6)
        admission_type_id = c2.selectbox(
            "Admission type", list(ADMISSION_TYPE.keys()),
            format_func=lambda x: ADMISSION_TYPE[x])
        admission_source_id = c3.selectbox(
            "Admission source", list(ADMISSION_SOURCE.keys()),
            format_func=lambda x: ADMISSION_SOURCE[x])

        discharge_disposition_id = st.selectbox(
            "Discharge disposition", list(DISCHARGE_DISPOSITION.keys()),
            format_func=lambda x: DISCHARGE_DISPOSITION[x])

        st.markdown('<div class="section-label">Clinical Utilization</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        time_in_hospital = c1.number_input("Time in hospital (days)", 1, 14, 3)
        num_lab_procedures = c2.number_input("Number of lab procedures", 0, 150, 40)
        num_procedures = c3.number_input("Number of procedures", 0, 10, 1)
        num_medications = c4.number_input("Number of medications", 0, 100, 12)

        c1, c2, c3, c4 = st.columns(4)
        number_outpatient = c1.number_input("Prior outpatient visits", 0, 50, 0)
        number_emergency = c2.number_input("Prior emergency visits", 0, 50, 0)
        number_inpatient = c3.number_input("Prior inpatient admissions", 0, 20, 0)
        number_diagnoses = c4.number_input("Number of diagnoses", 1, 20, 7)

        st.markdown('<div class="section-label">Diagnoses (grouped categories)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        diag_1 = c1.selectbox("Primary diagnosis", DIAG_CATEGORIES, index=0)
        diag_2 = c2.selectbox("Secondary diagnosis", DIAG_CATEGORIES, index=1)
        diag_3 = c3.selectbox("Additional diagnosis", DIAG_CATEGORIES, index=5)

        st.markdown('<div class="section-label">Lab Results</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        max_glu_serum = c1.selectbox("Glucose serum test", ["Not Tested", "Norm", ">200", ">300"])
        A1Cresult = c2.selectbox("A1C result", ["Not Tested", "Norm", ">7", ">8"])

        st.markdown('<div class="section-label">Medications</div>', unsafe_allow_html=True)
        st.caption("Status of each diabetes medication during this encounter")
        med_values = {}
        med_cols = st.columns(4)
        for i, med in enumerate(MEDICATION_COLUMNS):
            with med_cols[i % 4]:
                med_values[med] = st.selectbox(med, MED_STATUS_OPTIONS, index=0, key=f"med_{med}")

        c1, c2 = st.columns(2)
        change = c1.selectbox("Medication changed this encounter?", ["No", "Ch"])
        diabetesMed = c2.selectbox("Diabetes medication prescribed?", ["Yes", "No"])

        payer_code = st.selectbox("Payer code", ["Unknown", "MC", "HM", "SP", "BC", "CP", "UN"])
        medical_specialty = st.selectbox(
            "Admitting specialty",
            ["Unknown", "InternalMedicine", "Family/GeneralPractice",
             "Cardiology", "Surgery-General", "Emergency/Trauma"])

        submitted = st.form_submit_button("Assess Risk")

    if submitted:
        form_data = {
            "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "payer_code": payer_code,
            "medical_specialty": medical_specialty,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum,
            "A1Cresult": A1Cresult,
            "change": change,
            "diabetesMed": diabetesMed,
            **med_values,
        }

        X_input = build_model_input(form_data)
        risk_score = float(xgb_model.predict_proba(X_input)[:, 1][0])

        patient_context = {
            "risk_score": risk_score,
            "prior_admissions": number_inpatient,
            "num_medications": num_medications,
            "discharge_disposition": DISCHARGE_DISPOSITION[discharge_disposition_id],
            "a1c_level": "High" if A1Cresult in [">7", ">8"] else "Controlled",
            "time_in_hospital": time_in_hospital,
        }
        st.session_state.patient_context = patient_context
        st.session_state.chat_history = []

# ==========================================================================
# RISK RESULT (native bordered container — avoids unbalanced-div bugs)
# ==========================================================================
if st.session_state.patient_context:
    ctx = st.session_state.patient_context
    risk_level = get_risk_level(ctx["risk_score"])
    risk_class = {"High": "risk-high", "Moderate": "risk-moderate", "Low": "risk-low"}[risk_level]

    with st.container(border=True):
        st.markdown('<div class="card-accent"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="risk-eyebrow">30-Day Readmission Risk</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="risk-score-number">{ctx["risk_score"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="risk-badge {risk_class}">{risk_level} risk</span>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-label" style="margin-top:0;">Flags</div>', unsafe_allow_html=True)
            flags = get_risk_flags(ctx)
            if flags:
                for f in flags:
                    st.markdown(f'<div class="flag-item">{f}</div>', unsafe_allow_html=True)
            else:
                st.write("None triggered.")

        if st.button("Clear patient and ask general questions instead"):
            st.session_state.patient_context = None
            st.session_state.chat_history = []
            st.rerun()

# ==========================================================================
# ASK A QUESTION (native bordered container, pink circular send button)
# ==========================================================================
with st.container(border=True):
    st.markdown('<div class="card-accent"></div>', unsafe_allow_html=True)

    if st.session_state.patient_context:
        st.markdown('<div class="section-label" style="margin-top:0;">Ask About This Patient</div>', unsafe_allow_html=True)
        st.caption("Your question will be answered using this patient's risk profile.")
    else:
        st.markdown('<div class="section-label" style="margin-top:0;">Ask a Question</div>', unsafe_allow_html=True)
        st.caption("Get evidence-based answers from trusted clinical guidelines on type 2 diabetes.")

    for q, a in st.session_state.chat_history:
        st.markdown(f'<div class="qa-question">{q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="qa-answer">{a}</div>', unsafe_allow_html=True)

    query = ""
    with st.container(border=True, key="search_bar"):
        input_col, btn_col = st.columns([40, 1], gap="small")
        with input_col:
            query = st.text_input(
                "Your question",
                placeholder="e.g. What should be included in a discharge plan?",
                label_visibility="collapsed",
                key="question_box",
            )
        with btn_col:
            ask_clicked = st.button("↑", key="ask_button")

if ask_clicked and query:
    if not hf_token:
        st.error("Please enter your Hugging Face token in the sidebar first.")
    else:
        with st.spinner("Retrieving guidelines and generating answer..."):
            answer = generate_answer(query, st.session_state.patient_context, hf_token)
        st.session_state.chat_history.append((query, answer))
        st.rerun()
