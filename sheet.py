import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# 1. Initialize State
if "name_val" not in st.session_state:
    st.session_state.name_val = ""
if "email_val" not in st.session_state:
    st.session_state.email_val = ""
if "msg_val" not in st.session_state:
    st.session_state.msg_val = ""

# APPS_SCRIPT_URL = st.secrets["APPS_SCRIPT_URL"]
load_dotenv()
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")

st.set_page_config(page_title="Contact Form", page_icon="📋")
st.title("📋 Contact Form")

# 2. Define the Submission Logic
def handle_submission():
    # Grab values from state (where the widgets saved them)
    n = st.session_state.name_input
    e = st.session_state.email_input
    m = st.session_state.msg_input

    if not n or not e or not m:
        st.error("Please fill in all fields.")
        return

    payload = {"name": n, "email": e, "message": m}
    
    try:
        response = requests.post(
            APPS_SCRIPT_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        result = response.json()

        if result.get("status") == "success":
            st.success("✅ Response saved!")
            st.balloons()
            # 3. Reset the state values tied to the widgets
            st.session_state.name_input = ""
            st.session_state.email_input = ""
            st.session_state.msg_input = ""
        else:
            st.error(f"Error: {result.get('message')}")
    except Exception as ex:
        st.error(f"Failed to connect: {ex}")

# 4. The Form
with st.form("contact_form"):
    # Use 'key' to automatically link widget to st.session_state
    st.text_input("Full Name", placeholder="John Doe", key="name_input")
    st.text_input("Email Address", placeholder="john@example.com", key="email_input")
    st.text_area("Message", placeholder="Your message here...", key="msg_input")

    # On_click calls our logic and handles the state reset safely
    st.form_submit_button("Submit", on_click=handle_submission)
