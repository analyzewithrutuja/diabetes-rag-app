"""
Hospital Readmission Risk + Diabetes Guideline RAG Assistant
================================================================
A Streamlit app combining:
1. Full patient intake form (all relevant columns from diabetic_data.xlsx)
2. XGBoost readmission risk prediction
3. RAG-based, personalized, citation-backed Q&A over ADA/WHO/CDC/Mayo
   guidelines, iterable for multiple questions per patient

Deployment target: Hugging Face Spaces (Streamlit SDK)

Required files in the same folder as this app.py:
- xgb_readmission_model.pkl
- model_columns.pkl
- chroma_db/               (the persisted Chroma vector database folder)
- requirements.txt
"""

import re
import random
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ==========================================================================
# PAGE CONFIG
# ==========================================================================
st.set_page_config(page_title="Diabetes Readmission Risk + Guideline Assistant",
                    layout="wide")

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
# REFERENCE VALUE LISTS (from the UCI Diabetes 130-US Hospitals codebook)
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
# PREPROCESSING — builds a model-ready row from form inputs
# ==========================================================================
def build_model_input(form_data: dict) -> pd.DataFrame:
    """Takes the raw form dict and turns it into a one-hot-encoded row
    aligned to the exact columns the trained model expects."""
    row = form_data.copy()
    df = pd.DataFrame([row])
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

def build_prompt(query: str, patient_context: dict | None) -> str:
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

    return f"""You are a clinical decision-support assistant. Answer the question
below using ONLY the guideline excerpts provided. If patient context is given,
tailor your answer to it and prioritize any flagged concerns. Cite the source
organization (e.g., ADA, WHO, CDC, Mayo) for each recommendation. If the
guidelines don't address something, say so rather than guessing.

{patient_block}

RELEVANT GUIDELINES:
{retrieved_context}

QUESTION:
{query}

Provide a clear, structured answer with source citations. Include a brief
disclaimer that this is a decision-support suggestion, not a replacement
for clinical judgment."""

def generate_answer(query: str, patient_context: dict | None, hf_token: str) -> str:
    client_llm = InferenceClient(token=hf_token)
    prompt = build_prompt(query, patient_context)
    response = client_llm.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==========================================================================
# SIDEBAR — API token
# ==========================================================================
st.sidebar.title("Settings")
hf_token = st.sidebar.text_input("Hugging Face API token", type="password")
if not hf_token:
    st.sidebar.warning("Enter your Hugging Face token to enable answers.")

# ==========================================================================
# SESSION STATE
# ==========================================================================
if "patient_context" not in st.session_state:
    st.session_state.patient_context = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "general_chat_history" not in st.session_state:
    st.session_state.general_chat_history = []

# ==========================================================================
# TABS
# ==========================================================================
tab1, tab2 = st.tabs(["🩺 New Patient Assessment", "💬 Ask the Guidelines"])

# --------------------------------------------------------------------------
# TAB 1: Patient intake form + risk + iterative Q&A
# --------------------------------------------------------------------------
with tab1:
    st.header("New Patient Assessment")

    with st.form("patient_form"):
        st.caption(f"Auto-generated Encounter ID: {random.randint(100000, 999999)} "
                   f"(display only — not used in prediction)")

        st.subheader("Demographics & Admission")
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

        st.subheader("Clinical Utilization")
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

        st.subheader("Diagnoses (grouped categories)")
        c1, c2, c3 = st.columns(3)
        diag_1 = c1.selectbox("Primary diagnosis", DIAG_CATEGORIES, index=0)
        diag_2 = c2.selectbox("Secondary diagnosis", DIAG_CATEGORIES, index=1)
        diag_3 = c3.selectbox("Additional diagnosis", DIAG_CATEGORIES, index=5)

        st.subheader("Lab Results")
        c1, c2 = st.columns(2)
        max_glu_serum = c1.selectbox("Glucose serum test", ["Not Tested", "Norm", ">200", ">300"])
        A1Cresult = c2.selectbox("A1C result", ["Not Tested", "Norm", ">7", ">8"])

        st.subheader("Medications")
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
        st.session_state.chat_history = []  # reset Q&A for the new patient

    # ---- Display risk result ----
    if st.session_state.patient_context:
        ctx = st.session_state.patient_context
        risk_level = get_risk_level(ctx["risk_score"])
        color = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}[risk_level]

        st.divider()
        st.subheader("Risk Assessment Result")
        c1, c2 = st.columns([1, 2])
        c1.metric("30-Day Readmission Risk", f"{ctx['risk_score']*100:.1f}%", risk_level)
        c1.write(f"{color} **{risk_level} risk**")

        flags = get_risk_flags(ctx)
        with c2:
            st.write("**Flags:**")
            if flags:
                for f in flags:
                    st.write(f"- {f}")
            else:
                st.write("None triggered.")

        st.divider()
        st.subheader("Ask About This Patient")
        st.caption("Ask as many questions as you like — each builds on this patient's profile.")

        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        patient_query = st.chat_input("e.g. What should the discharge plan include?")
        if patient_query:
            if not hf_token:
                st.error("Please enter your Hugging Face token in the sidebar first.")
            else:
                with st.spinner("Retrieving guidelines and generating answer..."):
                    answer = generate_answer(patient_query, ctx, hf_token)
                st.session_state.chat_history.append((patient_query, answer))
                st.rerun()

# --------------------------------------------------------------------------
# TAB 2: General guideline Q&A (no patient context)
# --------------------------------------------------------------------------
with tab2:
    st.header("Ask the Diabetes Management Guidelines")
    st.caption("General questions answered from ADA, WHO, CDC, and Mayo Clinic guidelines — no patient context.")

    for q, a in st.session_state.general_chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    general_query = st.chat_input("e.g. What are the glycemic targets for hospitalized patients?", key="general_input")
    if general_query:
        if not hf_token:
            st.error("Please enter your Hugging Face token in the sidebar first.")
        else:
            with st.spinner("Retrieving guidelines and generating answer..."):
                answer = generate_answer(general_query, None, hf_token)
            st.session_state.general_chat_history.append((general_query, answer))
            st.rerun()
