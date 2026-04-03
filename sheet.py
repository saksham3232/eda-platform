import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")
APPS_SCRIPT_URL = st.secrets["APPS_SCRIPT_URL"]

st.set_page_config(page_title="Contact Form", page_icon="📋")
st.title("📋 Contact Form")
st.markdown("Fill out the form below. Your response is saved to Google Sheets.")

with st.form("contact_form"):
    name    = st.text_input("Full Name",     placeholder="John Doe")
    email   = st.text_input("Email Address", placeholder="john@example.com")
    message = st.text_area("Message",        placeholder="Your message here...")
    
    submitted = st.form_submit_button("Submit")

if submitted:
    if not name or not email or not message:
        st.error("Please fill in all fields.")
    else:
        payload = {"name": name, "email": email, "message": message}

        with st.spinner("Saving your response..."):
            try:
                response = requests.post(
                    APPS_SCRIPT_URL,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"}
                )
                result = response.json()

                if result.get("status") == "success":
                    st.success("✅ Response saved to Google Sheets!")
                    st.balloons()
                else:
                    st.error(f"Error: {result.get('message')}")

            except Exception as e:
                st.error(f"Failed to connect: {e}")