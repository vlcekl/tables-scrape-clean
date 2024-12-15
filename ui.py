import streamlit as st
import pandas as pd
from io_tables import load_scraped_data, save_harvested_tables
from extract_tables import harvest_tables_with_context
from process_tables import load_initial_prompt, query_llm, cleanup_table, create_prompt

# Load previously scraped raw data into session_state
if "sources_data" not in st.session_state:
    st.session_state["sources_data"] = load_scraped_data()

if "cleaned_data_store" not in st.session_state:
    st.session_state["cleaned_data_store"] = {}

# Streamlit app design
title = "Table Combine"
st.set_page_config(layout="wide", page_title=title)
st.title(title)

# Sidebar controls
st.sidebar.header("Controls")

# Load or scrape option
mode = st.sidebar.radio("Mode:", ["Scrape New", "Load Raw", "Load Clean"])

source_data = {"source": '', "tables": []}

if mode == "Scrape New":
    source_name = st.sidebar.text_input("Enter URL to scrape:", "")
    if source_name:
        st.sidebar.write("Scraping the website...")
        source_data = harvest_tables_with_context(source_name)
        st.session_state["sources_data"][source_name] = source_data  # Add to sources_data for raw data selection
        save_harvested_tables(source_data)
elif mode == "Load Raw":
    if st.session_state["sources_data"]:
        source_name = st.sidebar.selectbox("Select Source (Raw Data):", list(st.session_state["sources_data"].keys()))
        source_data = st.session_state["sources_data"][source_name]
elif mode == "Load Clean":
    if st.session_state["cleaned_data_store"]:
        source_name = st.sidebar.selectbox("Select Source (Clean Data):", list(st.session_state["cleaned_data_store"].keys()))
        source_data = st.session_state["cleaned_data_store"][source_name]
    else:
        st.sidebar.write("No cleaned data available.")
        source_data["New Table"] = pd.DataFrame()

# Table selection
if source_data["tables"]:
    selected_table = st.sidebar.selectbox("Select Table #", list(range(len(source_data["tables"]))))
else:
    selected_table = None


# Process button
process_button = st.sidebar.button("Process")

# Main content layout
st.subheader("Raw Data")

# Raw Data Schema
with st.expander("Source Info"):
    if source_name in st.session_state['sources_data']:
        st.write(f"Num tables: {len(source_data['tables'])}")

with st.expander("Tables Info"):
    st.write([s['context'] for s in source_data["tables"]])

# Raw Data Table
if selected_table is not None:
    with st.expander(f"Table {selected_table}"):
        st.dataframe(source_data['tables'][selected_table]['data_frame'])

if process_button and selected_table is not None:
    st.subheader("Cleaned Data")

    df_precleaned = cleanup_table(source_data['tables'][selected_table]['data_frame'])

    # Pre-cleaned data frame
    with st.expander(f"Pre-cleaned Table {selected_table}"):
        st.dataframe(df_precleaned)

    with st.expander(f"Context: {selected_table}"):
        st.write(source_data['tables'][selected_table]['context'])

    # Compile a prompt
    prompt_start = load_initial_prompt()
    prompt = create_prompt(df_precleaned, source_data['tables'][selected_table]['context'], prompt_start)
    with st.expander(f"Prompt: {selected_table}"):
        st.write(prompt)

    # Query LLM - print response
    response = query_llm(prompt)
    with st.expander(f"Response: {selected_table}"):
        st.write(response)
