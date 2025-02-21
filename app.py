import streamlit as st

st.title("GitHub Actions Streamlit Demo")

# Add text input
user_input = st.text_input("Enter your name:", "")

if st.button("Submit"):
    st.write(f"Hello, {user_input}!")
