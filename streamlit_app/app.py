"""Optional Streamlit version. Run from project root:
   streamlit run streamlit_app/app.py
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL = Path(__file__).resolve().parent.parent / "backend" / "model" / "model.pkl"
model = pickle.load(open(MODEL, "rb"))

st.title("Student Course Purchase Prediction")
st.write("Enter student details to predict whether they will purchase the course.")

age = st.number_input("Age", 10, 60, 20)
hours = st.number_input("Study Hours Per Week", 0, 50, 10)
courses = st.number_input("Previous Courses Completed", 0, 20, 2)
visits = st.number_input("Platform Visits Per Month", 0, 100, 15)
rate = st.number_input("Assignment Completion Rate (%)", 0, 100, 80)

if st.button("Predict Purchase"):
    row = pd.DataFrame([[age, hours, courses, visits, rate]],
                       columns=list(model.feature_names_in_))
    prob = model.predict_proba(row)[0][1]
    if prob >= 0.5:
        st.success(f"Likely to purchase ({prob:.0%} probability)")
    else:
        st.error(f"Not likely to purchase ({prob:.0%} probability)")
