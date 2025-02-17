import ollama
import streamlit as st

st.title("Embed Text")

if text := st.chat_input("Text to embed"):
    st.write(embedding := ollama.embed(model='bge-m3', input=text).embeddings)