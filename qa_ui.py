import ollama
import streamlit as st
from process_tables import TableMetadata

st.title("Question - Answer")
st.write(TableMetadata.model_json_schema())
if prompt := st.chat_input("User prompt"):
    with st.chat_message("Question:"):
        st.markdown(prompt)
    with st.chat_message("Answer:"):
        response = ollama.generate(model='llama3.2', prompt=prompt, format=TableMetadata.model_json_schema()).response
        st.markdown(response)

