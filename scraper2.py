import streamlit as st
import pandas as pd
import requests
import os
import hashlib
import pickle
from io import BytesIO
from bs4 import BeautifulSoup
#from .scrape_tables import scrape_tables

class WebScraperApp:
    def __init__(self):
        self.data_store = self.load_data()
        self.setup_page()

    def setup_page(self):
        st.set_page_config(page_title="HI Web Scraper & Tabular Data Processor with LLM", layout="wide")
        st.title("Web Scraper & Tabular Data Processor using Local LLM")
        st.markdown("Scrape, clean, and save tables from the web.")

    def save_data(self, key, data):
        if not os.path.exists("data"):
            os.makedirs("data")
        with open(f"data/{key}.pkl", "wb") as f:
            pickle.dump(data, f)

    def load_data(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        data_dict = {}
        for file in os.listdir("data"):
            if file.endswith('.pkl'):
                key = file.replace('.pkl', '')
                with open(f"data/{file}", "rb") as f:
                    data_dict[key] = pickle.load(f)
        return data_dict

    def choose_action(self):
        st.sidebar.header("Input Section")
        return st.sidebar.radio("Step 1: Choose Action", ("Process a New Website", "Load Previously Processed Data"))

    def input_url_section(self):
        return st.sidebar.text_input("Step 2: Enter URL to Scrape")

    def select_data_format(self):
        return st.sidebar.radio("Step 3: Select Data Format", ["HTML Tables", "Excel Files"])

    def scrape_data(self, url, data_format):
        if url and data_format:
            try:
                response = requests.get(url)
                response.raise_for_status()
                st.sidebar.success("Data fetched successfully!")
                st.write("URL: ", response.url)
                st.sidebar.write("URL: ", response.url)
                return response
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"An error occurred: {e}")
        return None

    def process_data(self, response, data_format):
        if response:
            url_hash = hashlib.md5(response.url.encode()).hexdigest()
            soup = BeautifulSoup(response.text, 'html.parser')
            if data_format == "HTML Tables":
                tables = soup.find_all('table')
                if tables:
                    table_index = st.sidebar.selectbox("Select table index:", list(range(len(tables))))
                    df = pd.read_html(str(tables[table_index]))[0]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [' '.join(col).strip() for col in df.columns.values]
                    self.save_processed_data(url_hash, response.url, df)
                    return df
            elif data_format == "Excel Files":
                links = [link['href'] for link in soup.find_all('a', href=True) if link['href'].endswith(('.xls', '.xlsx'))]
                if links:
                    excel_index = st.sidebar.selectbox("Select Excel file index:", list(range(len(links))))
                    excel_url = requests.compat.urljoin(response.url, links[excel_index])
                    try:
                        excel_response = requests.get(excel_url)
                        excel_response.raise_for_status()
                        df = pd.read_excel(BytesIO(excel_response.content))
                        self.save_processed_data(url_hash, response.url, df)
                        return df
                    except requests.exceptions.RequestException as e:
                        st.sidebar.error(f"Failed to download the Excel file: {e}")
        return None

    def save_processed_data(self, url_hash, url, df):
        if url_hash not in self.data_store:
            self.data_store[url_hash] = {'url': url, 'tables': []}
        self.data_store[url_hash]['tables'].append(df)
        self.save_data(url_hash, self.data_store[url_hash])

    def browse_data(self, df):
        if df is not None:
            selected_columns = st.sidebar.multiselect("Select columns to visualize:", df.columns.tolist(), default=df.columns.tolist())
            st.write("### Selected Columns")
            st.dataframe(df[selected_columns])

    def interact_with_llm(self, df):
        if df is not None:
            query = st.sidebar.text_area("Enter your query for the LLM regarding the data:")
            feedback = st.sidebar.text_area("Provide feedback or adjustments for the data:")
            if st.sidebar.button("Submit Query"):
                try:
                    headers = {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
                    data = {"prompt": query + "\nFeedback: " + feedback, "context": df.to_csv()}
                    response = requests.post("http://localhost:11434/api/ask", headers=headers, json=data)
                    response.raise_for_status()
                    st.write("### Response from LLM:")
                    st.write(response.json().get("response"))
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"Failed to interact with LLM: {e}")

    def browse_saved_data(self):
        urls = [self.data_store[key]['url'] for key in self.data_store]
        if urls:
            selected_url = st.sidebar.selectbox("Select a website to load data from:", urls)
            selected_key = [key for key in self.data_store if self.data_store[key]['url'] == selected_url][0]
            tables = self.data_store[selected_key]['tables']
            table_index = st.sidebar.selectbox("Select a table to load:", list(range(len(tables))))
            df = tables[table_index]
            st.write(f"### Previously Processed Data from: {selected_url}, Table {table_index}")
            st.dataframe(df)
            return df
        else:
            st.sidebar.text("No previously processed data available.")
        return None

    def main(self):
        action = self.choose_action()
        df = None
        if action == "Process a New Website":
            url = self.input_url_section()
            data_format = self.select_data_format()
            if url and data_format:
                response = self.scrape_data(url, data_format)
                df = self.process_data(response, data_format)
        else:
            df = self.browse_saved_data()
        self.browse_data(df)
        self.interact_with_llm(df)

if __name__ == "__main__":
    app = WebScraperApp()
    app.main()
