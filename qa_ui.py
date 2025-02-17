import ollama
import streamlit as st
from process_context import ContextSummary

def get_system_prompt():
    return "Provide a brief structured summary of the following text. Limit text description to roughly 5 sentences and include price or production trends for all produce and crops. Try to identify time frame and geography (country, region, etc.).\n\n#TEXT\n"

st.title("Question - Answer")

# select a model
selected_model = st.selectbox("Select model", [m.model for m in ollama.list().models])

# Ask question, show answer
if prompt := st.chat_input("User prompt"):
    prompt = get_system_prompt() + prompt
    with st.chat_message("Question:"):
        st.markdown(prompt)
    with st.chat_message("Answer:"):
        response = ollama.generate(model=selected_model, prompt=prompt, format=ContextSummary.model_json_schema(), options={'temperature': 0}).response
        st.markdown(response)

