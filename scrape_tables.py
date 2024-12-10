import streamlit as st
import requests
import os
import pickle
from .extract_tables import harvest_tables_with_context
from .process_tables import query_llm

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

    def input_url_to_scrape(self):
        return st.sidebar.text_input("Step 2: Enter URL to Scrape", key="input_url")

    def select_scraped_url(self):

        urls = [self.data_store[key]['source'] for key in self.data_store]
        if not urls:
            st.sidebar.text("No previously processed data available.")
            return None

        selected_url = st.sidebar.selectbox("Select a website to load data from:", urls, key="select_website")

        selected_key = [key for key in self.data_store if self.data_store[key]['source'] == selected_url][0]

        return self.data_store[selected_key]


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

    def interact_with_llm(self, table):

        output_table = query_llm(table)

        query = st.sidebar.text_area("Enter your query for the LLM regarding the data:", key="query")
        feedback = st.sidebar.text_area("Provide feedback or adjustments for the data:", key="feedback")

        if st.sidebar.button("Submit Query", key="submit_query"):
            try:
                headers = {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
                data = {"prompt": query + "\nFeedback: " + feedback, "context": df.to_csv()}
                response = requests.post("http://localhost:11433/api/ask", headers=headers, json=data)
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
            url = self.input_url_to_scrape()
            if url is not None and st.sidebar.button("Start Processing", key="start_processing"):

                tables = harvest_tables_with_context(url)

                if tables:
                    st.sidebar.success("Data fetched successfully!")
                    # Save all harvested tables along with metadata
                    self.save_harvested_tables(tables)
                else:
                    st.sidebar.warning("No data fetched")

                st.session_state['tables'] = tables

        elif action == "Load Previously Scraped Data":
            st.session_state['tables'] = self.select_scraped_url()

        if st.session_state['tables']:

            # Select a table to present
            table = self.select_harvested_table(st.session_state['tables'])

            # Process tables
            table = self.prepare_table(table)
            self.show_table(table)
            self.inspect_table(table)
            self.interact_with_llm(table)

if __name__ == "__main__":
    app = WebScraperApp()
    app.main()