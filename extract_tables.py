import os
import re
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag
import pandas as pd

def extract_tables_with_context(html_content, base_url):

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all tables in the HTML content
    tables = soup.find_all('table')

    # List to hold extracted table data and context
    extracted_tables = []

    for table in tables:

        df = pd.read_html(str(table))[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]

        # Initialize context information
        table_context = {
            'context': {
                'caption': '',
                'preceding_heading': '',
                'surrounding_text': '',
            },
            'headers': [],
            'data_rows': [],
            'data_frame': df
        }

        # Extract table caption if available
        caption = table.find('caption')
        if caption:
            table_context['context']['caption'] = caption.get_text(strip=True)
        
        # Extract preceding heading (h1 - h6 tags)
        heading = None
        for tag in table.find_all_previous():
            if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                heading = tag.get_text(strip=True)
                break
        if heading:
            table_context['context']['preceding_heading'] = heading

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
        for row_idx, row in enumerate(table.find_all('tr')):
            if row.find_all('td'):
                cells = []
                for col_idx, cell in enumerate(row.find_all('td')):
                    cell_text = cell.get_text(strip=True)
                    cells.append({'text': cell_text, 'row': row_idx, 'col': col_idx})
                data_rows.append(cells)
        table_context['data_rows'] = data_rows

        # Append the extracted table context to the list
        extracted_tables.append(table_context)

        # Find and extract tables from linked Excel files within the table
        excel_links = table.find_all('a', href=re.compile(r'\.xlsx?$'))
        for link in excel_links:
            excel_url = urljoin(base_url, link.get('href'))
            excel_filename = link.get('href').split('/')[-1]
            try:
                # Locate the cell containing the Excel link
                link_cell = link.find_parent('td')
                row_name = ''
                col_name = ''
                if link_cell:
                    row = link_cell.find_parent('tr')
                    if row:
                        row_idx = table.find_all('tr').index(row)
                        col_idx = row.find_all('td').index(link_cell)
                        # Extract row name and column name from context
                        row_name = table_context['data_rows'][row_idx][0]['text'] if row_idx < len(table_context['data_rows']) and len(table_context['data_rows'][row_idx]) > 0 else ''
                        col_name = table_context['headers'][0][col_idx]['text'] if len(table_context['headers']) > 0 and col_idx < len(table_context['headers'][0]) else ''

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
                        'context': {
                            'excel_filename': excel_filename,
                            'caption': f'Extracted from Excel: {sheet_name}',
                            'preceding_heading': table_context['context']['preceding_heading'],
                            'surrounding_text': table_context['context']['surrounding_text'],
                            'html_table_row_name': row_name,
                            'html_table_col_name': col_name,
                        },
                        'headers': [headers],
                        'data_rows': data_rows,
                        'data_frame': df
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

def scrape_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return None


def harvest_tables_with_context(url):

    response = scrape_url(url)

    if not response:
        return None

    tables_with_context = extract_tables_with_context(response.content, url)

    return {"source": url, "tables": tables_with_context}


# Example usage:
if __name__ == '__main__':
    # URL of the webpage to scrape
    url = 'https://example.com/page-with-tables'
    url = 'https://croptesting.iastate.edu/Corn/CornDistrict2.aspx'
    url = 'https://vt.cropsci.illinois.edu/corn/'

    # Extract tables with context
    tables = harvest_tables_with_context(url)
    tables_with_context = tables['tables']

    # Print extracted information
    for idx, table_info in enumerate(tables_with_context):
        print(f"Table {idx + 1}:")
        print(f"Caption: {table_info['context']['caption']}")
        print(f"Preceding Heading: {table_info['context']['preceding_heading']}")
        print(f"Surrounding Text: {table_info['context']['surrounding_text']}")
        print("Headers:")
        for header_row in table_info['headers']:
            print([cell['text'] for cell in header_row])
        print("Data Rows:")
        for data_row in table_info['data_rows']:
            if isinstance(data_row, list) and all(isinstance(cell, dict) for cell in data_row):
                print([cell['text'] for cell in data_row])
            else:
                print(data_row)
        if 'html_table_row_name' in table_info['context']:
            print(f"HTML Table Row Name: {table_info['context']['html_table_row_name']}, Column Name: {table_info['context']['html_table_col_name']}")
        if 'excel_filename' in table_info['context']:
            print(f"Excel Filename: {table_info['context']['excel_filename']}")
        print("\n" + "-" * 50 + "\n")
