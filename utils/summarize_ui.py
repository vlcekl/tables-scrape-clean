import ollama
import streamlit as st

def get_system_prompt():
    return "Your task is to summarize the provided text below. Please carefully identify significant entities and relationships and formulate a brief summary that captures them. You will be evaluated on the accuracy and brevity of the summary.\n\n # TEXT\n\n"

st.title("Summarize text")

# select a model
selected_model = st.selectbox("Select model", [m.model for m in ollama.list().models])

# Ask question, show answer
if prompt := st.chat_input("User prompt"):
    prompt = get_system_prompt() + prompt
    with st.chat_message("Question:"):
        st.markdown(prompt)
    with st.chat_message("Answer:"):
        #response = ollama.generate(model=selected_model, prompt=prompt, format=ContextSummary.model_json_schema(), options={'temperature': 0}).response
        response = ollama.generate(model=selected_model, prompt=prompt, options={'temperature': 0}).response
        st.markdown(response)
