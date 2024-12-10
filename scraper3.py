import streamlit as st
import pandas as pd
import requests
import os
import hashlib
import pickle
from io import BytesIO
from bs4 import BeautifulSoup

class WebScraperApp:
    def __init__(self):
        self.data_store = self.load_scraped_data()
        self.setup_page()

    # UI functions

    def setup_page(self):
        title = "Table Harvester"
        st.set_page_config(page_title=title, layout="wide")
        st.title(title)
        st.markdown("Scrape, clean, and save tables from the web.")

    def choose_action(self):
        st.sidebar.header("Input Section")
        return st.sidebar.radio("Step 1: Choose Action", ("Scrape a New Website", "Load Previously Scraped Data"), key="choose_action")

    def input_url_section(self):
        return st.sidebar.text_input("Step 2: Enter URL to Scrape", key="input_url")

    def select_data_format(self):
        return st.sidebar.radio("Step 3: Select Data Format", ["HTML Tables", "Excel Files"], index=None, key="data_format")

    def inspect_table(self, table):
        """Inspect a selected data frame"""
        df = table['data_processed']
        if df is not None:
            selected_columns = st.sidebar.multiselect("Select columns to visualize:", df.columns.tolist(), default=df.columns.tolist(), key="select_columns")
            st.write("### Selected Columns")
            st.dataframe(df[selected_columns])

    # IO functions

    def load_scraped_data(self):
        """Load all pickles from data directory. Return empty dict if no data found."""

        data_dict = {}
        if os.path.exists("data"):
            files = [file for file in os.listdir("data") if file.endswith('.pkl')]
            for file in files:
                key = file.replace('.pkl', '')
                with open(f"data/{file}", "rb") as f:
                    data_dict[key] = pickle.load(f)
        return data_dict

    def save_harvested_tables(self, tables):
        """Create a dict for data store and pickle it."""

        url_hash = tables['url_hash']
        self.data_store[url_hash] = tables

        if not os.path.exists("data"):
            os.makedirs("data")

        with open(f"data/{url_hash}.pkl", "wb") as f:
            pickle.dump(self.data_store[url_hash], f)

    # Data ingestion functions

    def scrape_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            st.sidebar.success("Data fetched successfully!")
            return response
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"An error occurred: {e}")

        return None

    def harvest_tables(self, url, data_format):

        response = self.scrape_url(url)

        if not response:
            return None

        url_hash = hashlib.md5(response.url.encode()).hexdigest()
        site = BeautifulSoup(response.text, 'html.parser')

        tables = []
        if data_format == "HTML Tables":
            html_tabs = site.find_all('table')
            for i, table in enumerate(html_tabs):
                df = pd.read_html(str(table))[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [' '.join(col).strip() for col in df.columns.values]
                tables.append({"data": df, "id": i, "metadata": {}})
        elif data_format == "Excel Files":
            links = [link['href'] for link in site.find_all('a', href=True) if link['href'].endswith(('.xls', '.xlsx'))]
            for i, link in enumerate(links):
                excel_url = requests.compat.urljoin(response.url, link)
                try:
                    excel_response = requests.get(excel_url)
                    excel_response.raise_for_status()
                    df = pd.read_excel(BytesIO(excel_response.content))
                    tables.append({"data": df, "id": i, "metadata": {}})
                except requests.exceptions.RequestException as e:
                    pass
                    #st.sidebar.error(f"{e}\nFailed to download the Excel file: {excel_url}.")

        return {"source": url, "url_hash": url_hash, "tables": tables}

    def select_scraped_url(self):

        urls = [self.data_store[key]['source'] for key in self.data_store]
        if not urls:
            st.sidebar.text("No previously processed data available.")
            return None

        selected_url = st.sidebar.selectbox("Select a website to load data from:", urls, key="select_website")

        selected_key = [key for key in self.data_store if self.data_store[key]['source'] == selected_url][0]

        return self.data_store[selected_key]

    def select_harvested_table(self, tables):

        # Default/initial index
        st.session_state['selected_table_index'] = 0

        table_index = st.sidebar.selectbox("Select a table to inspect:", list(range(len(tables['tables']))), index=st.session_state['selected_table_index'], key="table_index_selectbox")
        st.session_state['selected_table_index'] = table_index

        table = tables['tables'][table_index]

        return table

    def show_table(self, table):
        st.write(f"### Table {table['id']}")
        st.dataframe(table['data_processed'])

    # Processing functions

    def prepare_table(self, table):
        """Filter out columns and rows with all values missing"""

        df = table['data']
        print(df.info())
        df = df.dropna(axis='index', how='all')
        df = df.dropna(axis='columns', how='all')
        print('AFTER')
        print(df.info())

        table['data_processed'] = df

        return table

    def interact_with_llm(self, table):

        df = table['data_processed']

        if df is not None:
            query = st.sidebar.text_area("Enter your query for the LLM regarding the data:", key="query")
            feedback = st.sidebar.text_area("Provide feedback or adjustments for the data:", key="feedback")
            if st.sidebar.button("Submit Query", key="submit_query"):
                try:
                    headers = {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
                    data = {"prompt": query + "\nFeedback: " + feedback, "context": df.to_csv()}
                    response = requests.post("http://localhost:11434/api/ask", headers=headers, json=data)
                    response.raise_for_status()
                    st.write("### Response from LLM:")
                    st.write(response.json().get("response"))
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"Failed to interact with LLM: {e}")


    def main(self):

        action = self.choose_action()

        # Ingestion of scraped data either from web or local storage
        if 'tables' not in st.session_state:
            st.session_state['tables'] = None

        if action == "Scrape a New Website":
            # Scrape all tables from a website and, if successful, show the scraped results
            url = self.input_url_section()
            data_format = self.select_data_format()
            if url and data_format is not None and st.sidebar.button("Start Processing", key="start_processing"):
                tables = self.harvest_tables(url, data_format)
                # Save all harvested tables along with metadata
                if tables:
                    self.save_harvested_tables(tables)
                    st.session_state['tables'] = tables
        elif action == "Load Previously Scraped Data":
            st.session_state['tables'] = self.select_scraped_url()

        # Select a table to present
        table = None
        if st.session_state['tables']:
            table = self.select_harvested_table(st.session_state['tables'])

        # Process tables
        if table is not None:
            table = self.prepare_table(table)
            self.show_table(table)
            self.inspect_table(table)
            self.interact_with_llm(table)

if __name__ == "__main__":
    app = WebScraperApp()
    app.main()
