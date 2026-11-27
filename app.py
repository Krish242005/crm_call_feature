
import os
import json
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# 1. INITIAL CONFIGURATION
# -----------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
RESULTS_DIR = BASE_DIR / "results"

RECORDINGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Call Intelligence MVP",
    page_icon="📞",
    layout="wide"
)

st.title("📞 AI Call Intelligence")
st.caption("Recording → Transcription → AI Analysis → CRM Simulation")

if not API_KEY:
    st.error("OPENAI_API_KEY is missing. Check your .env file.")
    st.stop()

client = OpenAI(api_key=API_KEY)


# -----------------------------
# 2. AUDIO TRANSCRIPTION
# -----------------------------

def transcribe_audio(file_path):

    if Path(file_path).stat().st_size > 25 * 1024 * 1024:
        raise ValueError(
            "Audio exceeds 25 MB. Compress it or split it into smaller chunks."
        )

    with open(file_path, "rb") as audio:

        result = client.audio.transcriptions.create(
            model="gpt-transcribe",
            file=audio
        )

    return result.text


# -----------------------------
# 3. AI CALL ANALYSIS
# -----------------------------

def analyse_call(transcript):

    system_prompt = """
    You are an expert sales call intelligence analyst.

    Analyse the conversation and return valid JSON only.

    Do not invent information that is not present in the transcript.

    If information is unavailable, return null or an empty list.

    Return these fields:

    {
      "call_summary": "",
      "customer_intent": "",
      "sentiment": "",
      "keywords": [],
      "customer_requirements": {
        "product": null,
        "budget": null,
        "team_size": null,
        "timeline": null
      },
      "objections": [],
      "call_outcome": "",
      "suggested_crm_status": "",
      "next_action": "",
      "follow_up_date": null,
      "confidence": 0
    }

    Allowed CRM statuses:
    New Lead, Interested, Follow-up Required,
    Proposal Required, Demo Required,
    Not Interested, Converted, Pending Review.

    Confidence must be between 0 and 1.

    Do not assume a sale has happened merely because
    the customer expressed interest.
    """

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Analyse this call transcript:\n\n" + transcript
            }
        ],
        temperature=0.2
    )

    return json.loads(response.choices[0].message.content)


# -----------------------------
# 4. SAVE RESULTS
# -----------------------------

def save_result(record_id, data):

    output_path = RESULTS_DIR / f"{record_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return output_path


# -----------------------------
# 5. UPLOAD INTERFACE
# -----------------------------

st.header("Upload Call Recording")

uploaded_file = st.file_uploader(
    "Select an audio recording",
    type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm"]
)

customer_name = st.text_input("Customer Name")
agent_name = st.text_input("Salesperson Name")

if uploaded_file:

    st.audio(uploaded_file)

    st.write("File:", uploaded_file.name)
    st.write(
        "Size:",
        round(uploaded_file.size / (1024 * 1024), 2),
        "MB"
    )

    if st.button("Start AI Processing", type="primary"):

        record_id = str(uuid.uuid4())

        extension = Path(uploaded_file.name).suffix.lower()

        audio_path = RECORDINGS_DIR / f"{record_id}{extension}"

        audio_path.write_bytes(uploaded_file.getvalue())

        try:

            with st.status(
                "Processing call recording...",
                expanded=True
            ) as status:

                st.write("1. Recording stored successfully.")

                st.write("2. Converting audio into text...")

                transcript = transcribe_audio(audio_path)

                if not transcript.strip():
                    raise ValueError("No speech was detected.")

                st.write("3. Transcription completed.")

                st.write("4. Analysing customer conversation...")

                analysis = analyse_call(transcript)

                st.write("5. Generating CRM update preview...")

                result = {
                    "recording_id": record_id,
                    "customer_name": customer_name,
                    "agent_name": agent_name,
                    "recording_name": uploaded_file.name,
                    "created_at": datetime.now().isoformat(),
                    "transcript": transcript,
                    "analysis": analysis,
                    "crm_update": {
                        "status": analysis.get(
                            "suggested_crm_status",
                            "Pending Review"
                        ),
                        "next_action": analysis.get("next_action"),
                        "applied": False
                    }
                }

                save_result(record_id, result)

                status.update(
                    label="Processing completed!",
                    state="complete"
                )

            st.session_state["latest_result"] = result
            st.success("Call analysed successfully!")

        except Exception as error:

            st.error(f"Processing failed: {error}")


# -----------------------------
# 6. DISPLAY ANALYSIS
# -----------------------------

if "latest_result" in st.session_state:

    data = st.session_state["latest_result"]

    analysis = data["analysis"]

    st.divider()

    st.header("AI Analysis Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Sentiment",
        analysis.get("sentiment", "Unknown")
    )

    col2.metric(
        "Customer Intent",
        analysis.get("customer_intent", "Unknown")
    )

    col3.metric(
        "CRM Status",
        analysis.get("suggested_crm_status", "Pending Review")
    )

    st.subheader("Call Summary")

    st.write(analysis.get("call_summary"))

    st.subheader("Extracted Keywords")

    keywords = analysis.get("keywords", [])

    st.write(" | ".join(keywords))

    st.subheader("Customer Requirements")

    st.json(analysis.get("customer_requirements", {}))

    st.subheader("Objections")

    st.write(analysis.get("objections", []))

    st.subheader("Recommended Next Action")

    st.info(analysis.get("next_action", "No action identified"))

    st.subheader("Complete Transcript")

    st.text_area(
        "Transcript",
        data["transcript"],
        height=250
    )

    st.subheader("CRM Update Preview")

    st.json(data["crm_update"])

    st.download_button(
        "Download Complete Analysis (JSON)",
        data=json.dumps(data, indent=4, ensure_ascii=False),
        file_name=f"{data['recording_id']}.json",
        mime="application/json"
    )

