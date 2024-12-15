You are given raw tabular data and some metadata. Please analyze the data, identify any header rows and summary rows, and produce a cleaned-up schema with short but informative column names, data types, and short textual descriptions for each column.

### Raw Table Data
<PLACEHOLDER: RAW TABLE>

(The raw table may have multiple header rows or none, as well as possible summary rows. Each row may contain textual or numeric values.)

### Table Metadata
<PLACEHOLDER: METADATA>

(This metadata might have been scraped from the original website hosting the table. It may describe the table’s context, purpose, or potential meaning of columns.)

---

### Instructions

1. **Identify header rows** (could be one or multiple) and any summary row(s) in the raw data.  
2. **Classify each row** as header, data, or summary.  
3. **Infer a final table description**: a short sentence or two summarizing what the table represents.  
4. **Generate a JSON schema** describing each column in the cleaned-up version of the table. The schema should have short but informative column names (e.g. `"Region"`, `"Sales", `"Date"`, etc.), data types (`"string"`, `"float"`, `"int"`, `"date"`, etc.), and a short textual description of each column’s meaning.  
5. **Output a final JSON** object with the following structure (no extra keys or text outside of the JSON):
   ```jsonc
   {
     "table_description": "string - short description of what the table is about",
     "schema": [
       {
         "name": "short_column_name",
         "type": "data_type",
         "description": "short textual description"
       },
       ...
     ],
     "number_of_rows": {
        "header": number,
        "data": number,
        "summary": number
     }
   }
