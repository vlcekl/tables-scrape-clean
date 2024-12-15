You are provided with raw tabular data and associated metadata. Your task is to analyze this table, classify header rows vs. data rows vs. summary rows, then produce a JSON object that matches the 'TableSchemaOutput' Pydantic model. Do not include extra keys or text outside of the JSON. The fields in TableSchemaOutput are:

1. table_description (str): A short description of the table's purpose or content.
2. schema (List[ColumnSchema]): A list of columns, where each column has:
   - name (str): short but informative column name
   - type (str): a data type such as 'string', 'int', 'float', 'date'
   - description (str): brief description of what the column represents
3. header_rows_identified (int): number of rows identified as header
4. data_rows_identified (int): number of rows identified as data
5. summary_rows_identified (int): number of rows identified as summary

### Raw Table Data
<PLACEHOLDER: RAW TABLE>

(Provide the actual table text, CSV snippet, or messy multi-header table here.)

### Metadata Scraped from Website
<PLACEHOLDER: METADATA>

(This may describe table usage context, meaning of columns, notes about date ranges, etc.)

---

### Instructions for Output

1. **Analyze** the raw table. Identify any header row(s), data row(s), and summary row(s).
2. **Infer a final table description**: a short phrase or sentence summarizing what the table is about.
3. **Create a schema**: For each column, provide a `name` (short, readable), `type` (e.g. 'string', 'int', 'float', 'date'), and a short `description`.
4. **Populate** the integer fields: 
   - `header_rows_identified`
   - `data_rows_identified`
   - `summary_rows_identified`
   based on the classification of rows. If no headers or summary rows are present, set them to 0.
5. **Respond** with a **single JSON object** that exactly fits the `TableSchemaOutput` structure. 
   - No extra text or keys outside of that JSON object. 
   - No Markdown backticks.

**Note**: If unsure about a data type, default to `"string"`. If a column might contain numeric values, consider `"float"` or `"int"`. If you detect date formats, use `"date"`.

---
