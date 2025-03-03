import json
import pandas as pd
from enum import Enum
from pydantic import BaseModel, Field
import ollama


class DataType(str, Enum):
    """Data types identified in the raw tabular data column"""
    int_type = 'int'
    float_type = 'float'
    str_type = 'str'

class TableSchema(BaseModel):
    """Table schema: basic information"""
    column_name: str = Field(description="Short but informative column name derived from the header, data, and table description")
    data_type: DataType = Field(DataType.str_type, description="Identified data type - int, float, or str")
    description: str = Field(description="More informative description of a data column to be used by humans and LLMs.")

class RowCounts(BaseModel):
    """Number of taw table rows corresponding to header, data, and summary"""
    header: int = Field(0, description="Number of rows used to define a header")
    #data: int = Field(0, description="Number of rows corresponding to data")
    summary: int = Field(0, description="Number of rows corresponding to table summary statistics and footnotes")

class TableMetadata(BaseModel):
    """Table metadata describing raw tabular data,
      including table descriptions, header and summary rows numbers, and table schema"""
    table_description: str = Field(description="Informative description of the table based on the table context and data. To be used by humans and LLMs.")
    #header_rows_identified: int = Field(0, description="Identified number of header rows. Can be zero, one, or more")
    #data_rows_identified: int = Field(0, description="Identified number of data rows.")
    #summary_rows_identified: int = Field(0, description="Identified number of summary rows. Can be zero, one, or more")
    row_counts: RowCounts = Field(description="Identified numbers of header and summary rows")
    schema: list[TableSchema] = Field(description="Schema identified in raw tabular data")
    processing_info: str = Field(description="Information about raw data processing steps, such as how header, data, summary rows were identified, what context was used, and what choices were made.")

def create_tables_schema(dataset):

    tables_schema = []

    return tables_schema


def load_initial_prompt():

    #with open('./prompts/prompt_b.md', 'r') as f:
    with open('./prompts/prompt_json_simple.md', 'r') as f:
        prompt = f.read()

    return prompt

  # Processing functions

def cleanup_table(df):
    """Filter out columns and rows with all values missing"""

    df = df.dropna(axis='index', how='all')
    df = df.dropna(axis='columns', how='all')

    return df

def query_llm(prompt):

    return ollama.generate(model='llama3.2', prompt=prompt, format=TableMetadata.model_json_schema()).response

def create_prompt(df, context, prompt):

    # First part of the prompt
    full_prompt = prompt

#    # Add JSON with a table + context
#    full_prompt += "\n**Input Table (Raw)**\n"
#
#
#    full_prompt += "\n\n**Context Information**\n\n"
#
#    for k, v in context.items():
#        full_prompt += f"- **{k}**: {v}\n"

    # Move column names to first row
    df.loc[-1] = df.columns
    df.index = df.index + 1  # Shift the index
    df = df.sort_index()  # Sort the index

    # Add new column names (col1 ... coln)
    df.columns = [f"col{i}" for i in range(len(df.columns))]

    # Convert table to dictionary with new column names
    table_dict = df.to_dict(orient='records')

    input_dict = {
        "metadata": context,
        "table": table_dict
    }

    data_prompt = json.dumps(input_dict, indent=4, default=str)
    prompt = f"{prompt}\n```json\n{data_prompt}\n```"
    print(prompt)

    return prompt

def create_prompt_md(df, context, prompt):

    # First part of the prompt
    full_prompt = prompt

    # Add JSON with a table + context

    full_prompt += "\n\n**Context**\n\n"

    for k, v in context.items():
        full_prompt += f"- **{k}**: {v}\n"

    full_prompt += "\n**Table**\n"
    full_prompt += df.to_markdown(index=True)
    full_prompt += "\n"

    print(full_prompt)

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

