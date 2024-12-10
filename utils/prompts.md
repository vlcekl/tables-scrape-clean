Task: Please clean and organize the following table data extracted from a webpage.

Context:
Caption: [Caption Text]
Preceding Heading: [Heading Text]
Surrounding Text: [Surrounding Text]

Headers:
[
  [{"text": "Year", "colspan": 1, "rowspan": 2}, {"text": "Sales", "colspan": 2, "rowspan": 1}],
  [{"text": "Domestic", "colspan": 1, "rowspan": 1}, {"text": "International", "colspan": 1, "rowspan": 1}]
]

Headers: ["Year", "Sales - Domestic", "Sales - International"]

Data Rows:
[  ["2020", "5000", "3000"],
  ["2021", "5500", "3200"],
  ...
]


### Task:
Please clean and organize the following table data extracted from a webpage.

### Context:
Caption: [Caption Text]
Preceding Heading: [Heading Text]
Surrounding Text: [Surrounding Text]

### Headers:
[Headers List]

### Data Rows:
[Data Rows List]

### Instructions:
- Interpret the headers and align them with the data columns.
- Ensure data types are consistent (e.g., numbers, dates).
- Provide a brief description for each column.
- Output the cleaned table in CSV format.
- Include an overall schema at the end.



### Task:
Please clean and organize the following table data extracted from a webpage.

### Context:
Caption: Annual Sales Report
Preceding Heading: Company Financials
Surrounding Text: The table below shows the annual sales figures for the past two years.

### Headers:
[
  [{"text": "Year", "colspan": 1, "rowspan": 2}, {"text": "Sales", "colspan": 2, "rowspan": 1}],
  [{"text": "Domestic", "colspan": 1, "rowspan": 1}, {"text": "International", "colspan": 1, "rowspan": 1}]
]

### Data Rows:
[
  ["2020", "5000", "3000"],
  ["2021", "5500", "3200"]
]

### Instructions:
- Interpret the headers and align them with the data columns.
- Ensure data types are consistent (e.g., numbers, dates).
- Provide a brief description for each column.
- Output the cleaned table in CSV format.
- Include an overall schema at the end as a CSV table.
