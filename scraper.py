import streamlit as st
import pandas as pd
import requests
import os
import hashlib
import pickle
from io import StringIO, BytesIO
from bs4 import BeautifulSoup

# Setup basic configurations
def setup_page():
    st.set_page_config(
        page_title="Web Scraper & Tabular Data Processor with LLM",
        layout="wide"
    )
    st.title("Web Scraper & Tabular Data Processor using Local LLM")
    st.markdown("Scrape, clean, and save tables from the web.")

# Utility function to save and load data
def save_data(key, data):
    with open(f"data/{key}.pkl", "wb") as f:
        pickle.dump(data, f)

def load_data():
    if not os.path.exists("data"):
        os.makedirs("data")
    files = [f for f in os.listdir("data") if f.endswith('.pkl')]
    data_dict = {}
    for file in files:
        key = file.replace('.pkl', '')
        with open(f"data/{file}", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict

# Step 1: Choose to Process New Data or Load Existing Data
def choose_action():
    st.sidebar.header("Input Section")
    st.sidebar.subheader("Step 1: Choose Action")
    action = st.sidebar.radio("What would you like to do?", ("Process a New Website", "Load Previously Processed Data"))
    return action

# Step 2: Input URL for Web Scraping
def input_url_section():
    st.sidebar.subheader("Step 2: Enter URL to Scrape")
    url = st.sidebar.text_input("Enter the URL you want to scrape:")
    return url

# Step 3: Select Data Format to Scrape
def select_data_format():
    st.sidebar.subheader("Step 3: Select Data Format to Scrape")
    format_options = ["HTML Tables", "Excel Files"]
    selected_format = st.sidebar.radio("Select the format of data you want to scrape:", format_options)
    return selected_format

# Step 4: Web Scraping the Content
def scrape_data(url, data_format):
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            st.sidebar.success("Data fetched successfully!")
            return response, data_format
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"An error occurred: {e}")
            return None, None
    return None, None

# Step 5: Processing the scraped data to tabular form
def process_data(response, data_format):
    if response is not None:
        url_hash = hashlib.md5(response.url.encode()).hexdigest()
        if data_format == "HTML Tables":
            st.sidebar.subheader("Step 4: Process the Data")
            st.sidebar.text("Extracting HTML tables and processing them into DataFrames.")
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table')
            if tables:
                st.sidebar.write(f"Found {len(tables)} table(s) on the page. Select one to process:")
                table_index = st.sidebar.selectbox("Select table index:", list(range(len(tables))), key=f"select_table_{url_hash}")
                selected_table = tables[table_index]
                df = pd.read_html(str(selected_table))[0]
                # Handle multi-line headers
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [' '.join(col).strip() for col in df.columns.values]
                st.write("### Extracted Tabular Data")
                st.dataframe(df)
                st.write("### Data Schema")
                st.json({"columns": df.columns.tolist(), "dtypes": df.dtypes.apply(str).to_dict()})
                # Save processed data
                if url_hash not in data_store:
                    data_store[url_hash] = {'url': response.url, 'tables': []}
                data_store[url_hash]['tables'].append(df)
                save_data(url_hash, data_store[url_hash])
                return df
            else:
                st.sidebar.error("No tables found on the page.")
        elif data_format == "Excel Files":
            st.sidebar.subheader("Step 4: Process the Data")
            st.sidebar.text("Looking for Excel files linked on the page.")
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            excel_links = [link['href'] for link in links if link['href'].endswith(('.xls', '.xlsx'))]
            if excel_links:
                st.sidebar.write(f"Found {len(excel_links)} Excel file(s) on the page. Select one to process:")
                excel_index = st.sidebar.selectbox("Select Excel file index:", list(range(len(excel_links))), key=f"select_excel_{url_hash}")
                excel_url = excel_links[excel_index]
                if not excel_url.startswith("http"):
                    excel_url = requests.compat.urljoin(response.url, excel_url)
                try:
                    excel_response = requests.get(excel_url)
                    excel_response.raise_for_status()
                    df = pd.read_excel(BytesIO(excel_response.content))
                    st.write("### Extracted Tabular Data from Excel File")
                    st.dataframe(df)
                    st.write("### Data Schema")
                    st.json({"columns": df.columns.tolist(), "dtypes": df.dtypes.apply(str).to_dict()})
                    # Save processed data
                    if url_hash not in data_store:
                        data_store[url_hash] = {'url': response.url, 'tables': []}
                    data_store[url_hash]['tables'].append(df)
                    save_data(url_hash, data_store[url_hash])
                    return df
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"Failed to download the Excel file: {e}")
            else:
                st.sidebar.error("No Excel files found on the page.")
    return None

# Step 6: Browse and Visualize Tabular Data
def browse_data(df):
    if df is not None:
        st.sidebar.subheader("Step 6: Browse and Visualize the Data")
        st.sidebar.write("You can explore the tabular data below:")
        columns = df.columns.tolist()
        selected_columns = st.sidebar.multiselect("Select columns to visualize:", columns, default=columns)
        if selected_columns:
            st.write("### Selected Columns")
            st.dataframe(df[selected_columns])

# Step 7: Interacting with Local LLM (Ollama)
def interact_with_llm(df):
    st.sidebar.subheader("Step 7: Interact with Local LLM via Ollama")
    if df is not None:
        query = st.sidebar.text_area("Enter your query for the LLM regarding the data:")
        feedback = st.sidebar.text_area("Provide feedback or adjustments for the data:")
        if st.sidebar.button("Submit Query"):
            try:
                headers = {
                    "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
                }
                data = {
                    "prompt": query + "\nFeedback: " + feedback,
                    "context": df.to_csv()
                }
                # Placeholder URL - you need to replace this with the correct one for Ollama's local LLM API
                url = "http://localhost:11434/api/ask"
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                llm_response = response.json().get("response")
                st.write("### Response from LLM:")
                st.write(llm_response)
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Failed to interact with LLM: {e}")

# Step 8: Browse Previously Processed Data
def browse_saved_data():
    st.sidebar.subheader("Step 8: Browse Previously Processed Data")
    saved_data = load_data()
    urls = [saved_data[key]['url'] for key in saved_data] if saved_data else []
    selected_url = st.sidebar.selectbox("Select a website to load data from:", urls, key='browse_saved_website')
    if selected_url:
        selected_key = [key for key in saved_data if saved_data[key]['url'] == selected_url][0]
        tables = saved_data[selected_key]['tables']
        table_index = st.sidebar.selectbox("Select a table to load:", list(range(len(tables))), key='browse_saved_table')
        df = tables[table_index]
        st.write(f"### Previously Processed Data from: {selected_url}, Table {table_index}")
        st.dataframe(df)
        st.write("### Data Schema")
        st.json({"columns": df.columns.tolist(), "dtypes": df.dtypes.apply(str).to_dict()})
        return df
    else:
        st.sidebar.text("No previously processed data available.")
    return None

# Main function to run the Streamlit app
def main():
    global data_store
    data_store = load_data()
    setup_page()
    action = choose_action()

    if action == "Process a New Website":
        url = input_url_section()
        data_format = select_data_format()
        response, data_format = scrape_data(url, data_format)
        df = process_data(response, data_format)
    else:
        df = browse_saved_data()

    browse_data(df)
    interact_with_llm(df)

if __name__ == "__main__":
    main()
