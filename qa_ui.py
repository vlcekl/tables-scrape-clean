import ollama
import streamlit as st

st.title("Question - Answer")
if prompt := st.chat_input("User prompt"):
    with st.chat_message("Question:"):
        st.markdown(prompt)
    with st.chat_message("Answer:"):
        response = ollama.generate(model='llama3.2', prompt=prompt).response
        st.markdown(response)

