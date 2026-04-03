import streamlit as st
import requests
import json

# Paste your deployed Apps Script URL here
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbw1rh0x3MKfWOCLoQmQJ-XGog9qDRDC3lP8isdOMZCBn4zZQObOH83orCVFm0BCvb4e/exec"

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