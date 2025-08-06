import streamlit as st
from query_understanding import parse_query
from rule_based_decision_system import process_query

st.set_page_config(page_title="Policy Query System", layout="wide")
st.title("🧠 Insurance Query Assistant")

query = st.text_input("Enter your query (e.g., '46-year-old male, knee surgery in Pune, 3-month policy'):")

if st.button("Analyze Query"):
    with st.spinner("Processing..."):
        result = process_query(query)
        st.subheader("✅ Structured JSON Response")
        st.json(result)
