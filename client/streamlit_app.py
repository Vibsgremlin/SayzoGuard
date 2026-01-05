# Streamlit Admin UI


import streamlit as st
import requests
from leakage.file_extractor import extract_text_from_file

API = "http://localhost:8000/classify"

st.title("SayzoGuard - Leakage Detector")

mode = st.radio("Choose Input Type", ["Text", "File"])

if mode == "Text":
    txt = st.text_area("Enter message")
    task_id = st.text_input("Task ID (for escrow)")
    session_id = st.text_input("Session ID")

    if st.button("Analyze"):
        resp = requests.post(API, json={
            "text": txt,
            "task_id": task_id,
            "session_id": session_id
        }).json()
        st.json(resp)

else:
    file = st.file_uploader("Upload File", type=["png","jpg","jpeg","pdf","txt"])
    task_id = st.text_input("Task ID")
    session_id = st.text_input("Session ID")

    if file and st.button("Analyze"):
        text = extract_text_from_file(file)
        st.write("Extracted Text:")
        st.write(text)

        resp = requests.post(API, json={
            "text": text,
            "task_id": task_id,
            "session_id": session_id
        }).json()

        st.json(resp)
