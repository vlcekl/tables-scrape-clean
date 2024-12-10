import pandas as pd

raw_data = {
        "Website 0": {
            "tables_schema": pd.DataFrame({
                "id": [0, 2],
                "name": ["Table 0", "Table 2"],
                "description": ["Table 0 from Website 1", "Table 2 from Website 1"]
            }),
            "column_schema": pd.DataFrame({
                "column_name": ["Column A", "Column B", "Column X", "Column Y"],
                "table_name": ["Table 0", "Table 1", "Table 2", "Table 2"],
                "data_type": ["int", "str", "float", "bool"],
                "description": ["Integer values", "String values", "Floating-point values", "Boolean values"],
                "statistics": ["min: 0, max: 3", "unique: 3", "mean: 0.5", "True: 50%, False: 50%"]
            }),
            "Table 0": pd.DataFrame({
                "Column A": [0, 2, 3],
                "Column B": ["a", "b", "c"]
            }),
            "Table 1": pd.DataFrame({
                "Column X": [-1.1, 0.2, 0.3],
                "Column Y": [True, False, True]
            })
        },
        "Website 1": {
            "tables_schema": pd.DataFrame({
                "id": [0, 2],
                "name": ["Table A", "Table B"],
                "description": ["Table A from Website 1", "Table B from Website 2"]
            }),
            "column_schema": pd.DataFrame({
                "column_name": ["Column P", "Column Q", "Column R", "Column S"],
                "table_name": ["Table A", "Table A", "Table B", "Table B"],
                "data_type": ["int", "str", "float", "bool"],
                "description": ["Integer values", "String values", "Floating-point values", "Boolean values"],
                "statistics": ["min: 9, max: 30", "unique: 2", "mean: 5.5", "True: 80%, False: 20%"]
            }),
            "Table A": pd.DataFrame({
                "Column P": [9, 20, 30],
                "Column Q": ["x", "y", "z"]
            }),
            "Table B": pd.DataFrame({
                "Column R": [0.1, 2.2, 3.3],
                "Column S": [True, True, False]
            })
        },
        "Database 0": {
            "tables_schema": pd.DataFrame({
                "id": [0, 2],
                "name": ["Table X", "Table Y"],
                "description": ["Table X from Database 0", "Table Y from Database 1"]
            }),
            "column_schema": pd.DataFrame({
                "column_name": ["Column L", "Column M", "Column N", "Column O"],
                "table_name": ["Table X", "Table X", "Table Y", "Table Y"],
                "data_type": ["int", "str", "float", "bool"],
                "description": ["Integer values", "String values", "Floating-point values", "Boolean values"],
                "statistics": ["min: 4, max: 15", "unique: 4", "mean: 10.5", "True: 60%, False: 40%"]
            }),
            "Table X": pd.DataFrame({
                "Column L": [4, 10, 15],
                "Column M": ["alpha", "beta", "gamma"]
            }),
            "Table Y": pd.DataFrame({
                "Column N": [-1.5, 1.5, 2.5],
                "Column O": [False, True, False]
            })
        }
    }

def web_data(url):
    return {
        "tables_schema": pd.DataFrame({
            "id": [1],
            "name": ["New Table"],
            "description": [f"A newly scraped table from {url}"]
        }),
        "column_schema": pd.DataFrame({
            "column_name": ["Column Z"],
            "table_name": ["New Table"],
            "data_type": ["int"],
            "description": ["Dummy integer column"],
            "statistics": ["min: 0, max: 100"]
        }),
        "New Table": pd.DataFrame({
            "Column Z": [0, 50, 100]
        })
    }