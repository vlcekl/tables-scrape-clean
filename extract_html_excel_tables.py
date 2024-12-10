import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import re
import os
from urllib.parse import urljoin

def extract_tables_with_context(html_content, base_url):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all tables in the HTML content
    tables = soup.find_all('table')

    # List to hold extracted table data and context
    extracted_tables = []

    for table in tables:
        table_data = {}
        
        # Initialize context information
        table_context = {
            'caption': '',
            'preceding_heading': '',
            'surrounding_text': '',
            'headers': [],
            'data_rows': []
        }

        # Extract table caption if available
        caption = table.find('caption')
        if caption:
            table_context['caption'] = caption.get_text(strip=True)
        
        # Extract preceding heading (h1 - h6 tags)
        heading = None
        for tag in table.find_all_previous():
            if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                heading = tag.get_text(strip=True)
                break
        if heading:
            table_context['preceding_heading'] = heading

        # Extract surrounding text (previous and next sibling paragraphs)
        prev_paragraphs = []
        next_paragraphs = []

        # Previous siblings
        prev_sibling = table.previous_sibling
        while prev_sibling:
            if isinstance(prev_sibling, Tag) and prev_sibling.name == 'p':
                prev_paragraphs.insert(0, prev_sibling.get_text(strip=True))
                if len(prev_paragraphs) >= 2:
                    break
            prev_sibling = prev_sibling.previous_sibling

        # Next siblings
        next_sibling = table.next_sibling
        while next_sibling:
            if isinstance(next_sibling, Tag) and next_sibling.name == 'p':
                next_paragraphs.append(next_sibling.get_text(strip=True))
                if len(next_paragraphs) >= 2:
                    break
            next_sibling = next_sibling.next_sibling

        surrounding_text = ' '.join(prev_paragraphs + next_paragraphs)
        table_context['surrounding_text'] = surrounding_text

        # Extract table headers, including multi-level headers
        headers = []
        header_rows = table.find_all('tr')
        for row in header_rows:
            header_cells = row.find_all(['th'])
            if header_cells:
                header_row = []
                for cell in header_cells:
                    cell_text = cell.get_text(strip=True)
                    colspan = cell.get('colspan')
                    rowspan = cell.get('rowspan')
                    header_row.append({
                        'text': cell_text,
                        'colspan': int(colspan) if colspan else 1,
                        'rowspan': int(rowspan) if rowspan else 1
                    })
                headers.append(header_row)
            else:
                break  # Stop if no more header rows
        table_context['headers'] = headers

        # Extract table data rows
        data_rows = []
        for row in table.find_all('tr'):
            if row.find_all('td'):
                cells = []
                for cell in row.find_all('td'):
                    cell_text = cell.get_text(strip=True)
                    cells.append(cell_text)
                data_rows.append(cells)
        table_context['data_rows'] = data_rows

        # Append the extracted table context to the list
        extracted_tables.append(table_context)

    # Find and extract tables from linked Excel files
    excel_links = soup.find_all('a', href=re.compile(r'\.xlsx?$'))
    for link in excel_links:
        excel_url = urljoin(base_url, link.get('href'))
        try:
            # Download the Excel file
            response = requests.get(excel_url)
            response.raise_for_status()

            # Save the file temporarily
            with open('temp.xlsx', 'wb') as temp_file:
                temp_file.write(response.content)

            # Read the Excel file using Pandas
            xls = pd.ExcelFile('temp.xlsx')
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Convert the DataFrame to a list of lists (rows)
                headers = [{'text': col, 'colspan': 1, 'rowspan': 1} for col in df.columns]
                data_rows = df.values.tolist()

                # Create context for the Excel table
                excel_table_context = {
                    'caption': f'Extracted from Excel: {sheet_name}',
                    'preceding_heading': '',
                    'surrounding_text': f'Linked Excel file: {excel_url}',
                    'headers': [headers],
                    'data_rows': data_rows
                }

                # Append the extracted Excel table context to the list
                extracted_tables.append(excel_table_context)
        except requests.RequestException as e:
            print(f"Failed to download Excel file from {excel_url}: {e}")
        except Exception as e:
            print(f"Failed to process Excel file from {excel_url}: {e}")
        finally:
            # Clean up the temporary file if it exists
            if os.path.exists('temp.xlsx'):
                os.remove('temp.xlsx')

    return extracted_tables

# Example usage:
if __name__ == '__main__':
    # URL of the webpage to scrape
    url = 'https://example.com/page-with-tables'
    url = 'https://vt.cropsci.illinois.edu/corn/'

    # Fetch the HTML content
    response = requests.get(url)
    html_content = response.content

    # Extract tables with context
    tables_with_context = extract_tables_with_context(html_content, url)

    # Print extracted information
    for idx, table_info in enumerate(tables_with_context):
        print(f"Table {idx + 1}:")
        print(f"Caption: {table_info['caption']}")
        print(f"Preceding Heading: {table_info['preceding_heading']}")
        print(f"Surrounding Text: {table_info['surrounding_text']}")
        print("Headers:")
        for header_row in table_info['headers']:
            print([cell['text'] for cell in header_row])
        print("Data Rows:")
        for data_row in table_info['data_rows']:
            print(data_row)
        print("\n" + "-" * 50 + "\n")
