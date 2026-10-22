import streamlit as st
import requests
API_URL = "http://127.0.0.1:8000" # our FastAPI server
st.set_page_config(
page_title = "Student Predictor",
page_icon = "■",
layout = "centered"
)
st.title("■ Student Pass / Fail Predictor")
st.markdown("Enter the student's details below to predict their result.")
st.markdown("---")