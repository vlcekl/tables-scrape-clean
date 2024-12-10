import pandas as pd
import ollama

def create_tables_schema(dataset):

    tables_schema = []

    return tables_schema


def load_initial_prompt():

    with open('./prompts/prompt_initial_modified.md', 'r') as f:
        prompt = f.read()

    return prompt

  # Processing functions

def cleanup_table(df):
    """Filter out columns and rows with all values missing"""

    df = df.dropna(axis='index', how='all')
    df = df.dropna(axis='columns', how='all')

    return df

def query_llm(prompt):

    return ollama.generate(model='llama3.2', prompt=prompt).response

def create_prompt(df, context, prompt):

    full_prompt = prompt

    full_prompt += "\n**Input Table (Raw)**\n"

    full_prompt += df.to_markdown(index=False)

    full_prompt += "\n\n**Context Information**\n\n"

    for k, v in context.items():
        full_prompt += f"- **{k}**: {v}\n"

    return full_prompt

def clean_table_with_llm(raw_data, instructions):
    raw_data = raw_data.copy()
    raw_data[raw_data.columns[-1]] *= 2  # Example transformation
    return raw_data

def clean_table(df, table, prompt):
    """Ask LLM to cleanup and standardize the provided table"""

    # Remove empty rows and columns
    df = cleanup_table(df)

    full_instructions = create_prompt(df, context, prompt)

    return full_instructions

    # if df is not None:
    #     query = st.sidebar.text_area("Enter your query for the LLM regarding the data:", key="query")
    #     feedback = st.sidebar.text_area("Provide feedback or adjustments for the data:", key="feedback")
    #     if st.sidebar.button("Submit Query", key="submit_query"):
    #         try:
    #             headers = {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
    #             data = {"prompt": query + "\nFeedback: " + feedback, "context": df.to_csv()}
    #             response = requests.post("http://localhost:11433/api/ask", headers=headers, json=data)
    #             response.raise_for_status()
    #             st.write("### Response from LLM:")
    #             st.write(response.json().get("response"))
    #         except requests.exceptions.RequestException as e:
    #             st.sidebar.error(f"Failed to interact with LLM: {e}")

