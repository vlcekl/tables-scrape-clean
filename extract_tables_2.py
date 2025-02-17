import os
import re
import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
from urllib.parse import urljoin
from tabula import read_pdf
from PyPDF2 import PdfReader


def extract_tables_with_context(html_content, base_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_tables = []

    # Extract tables from HTML
    tables = soup.find_all('table')
    for table_idx, table in enumerate(tables):
        html_context = extract_html_context(table, table_idx)
        extracted_tables.append(extract_html_table_with_context(table, html_context))

    # Extract Excel and PDF tables from all links on the page
    links = soup.find_all('a', href=True)
    for link in links:
        if re.search(r'\.xlsx?$', link['href'], re.IGNORECASE):
            extracted_tables.extend(
                extract_excel_file(link, base_url, link_context_from_html_table(link, tables))
            )
        elif re.search(r'\.pdf$', link['href'], re.IGNORECASE):
            extracted_tables.extend(
                extract_pdf_file(link, base_url, link_context_from_html_table(link, tables))
            )

    return extracted_tables


def extract_html_context(table, table_idx):
    return {
        'caption': extract_table_caption(table),
        'preceding_heading': extract_preceding_heading(table),
        'surrounding_text': extract_surrounding_text(table),
        'html_table_reference': f'HTML Table {table_idx}'
    }


def extract_html_table_with_context(table, context):
    df = pd.read_html(str(table))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    headers = extract_table_headers(table)
    data_rows = extract_table_data_rows(table)

    return {
        'context': context,
        'headers': headers,
        'data_rows': data_rows,
        'data_frame': df
    }


def extract_table_caption(table):
    caption = table.find('caption')
    return caption.get_text(strip=True) if caption else ''


#def extract_preceding_heading(element):
#    for tag in element.find_all_previous():
#        if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
#            return tag.get_text(strip=True)
#    return ''


def extract_preceding_heading(element):
    """
    Extracts the full hierarchy of headings preceding the given element, 
    including the web page title as the root if available.

    Args:
        element: A BeautifulSoup element for which the heading hierarchy is to be extracted.

    Returns:
        str: A string representing the hierarchy of headings (e.g., "Page Title > Heading 1 > Subheading 1.1").
    """
    headings = []
    soup = element.find_parent('html')  # Get the parent HTML document

    # Add the page title as the root of the hierarchy, if available
    if soup and soup.title:
        page_title = soup.title.get_text(strip=True)
        if page_title:
            headings.append(page_title)

    # Traverse upwards through the DOM to collect headings
    current_element = element
    while current_element:
        for tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if current_element.name == tag_name:
                headings.append(current_element.get_text(strip=True))  # Add heading
                break
        current_element = current_element.find_previous()  # Move to the previous element in the DOM

    # Format the hierarchy as a string
    return ' > '.join(headings)


def extract_surrounding_text(element):
    prev_paragraphs, next_paragraphs = [], []

    prev_sibling = element.previous_sibling
    while prev_sibling:
        if isinstance(prev_sibling, Tag) and prev_sibling.name == 'p':
            prev_paragraphs.insert(0, prev_sibling.get_text(strip=True))
            if len(prev_paragraphs) >= 2:
                break
        prev_sibling = prev_sibling.previous_sibling

    next_sibling = element.next_sibling
    while next_sibling:
        if isinstance(next_sibling, Tag) and next_sibling.name == 'p':
            next_paragraphs.append(next_sibling.get_text(strip=True))
            if len(next_paragraphs) >= 2:
                break
        next_sibling = next_sibling.next_sibling

    return ' '.join(prev_paragraphs + next_paragraphs)


def extract_table_headers(table):
    headers = []
    header_rows = table.find_all('tr')
    for row in header_rows:
        header_cells = row.find_all(['th'])
        if header_cells:
            headers.append([cell.get_text(strip=True) for cell in header_cells])
        else:
            break
    return headers


def extract_table_data_rows(table):
    data_rows = []
    for row_idx, row in enumerate(table.find_all('tr')):
        if row.find_all('td'):
            cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
            data_rows.append(cells)
    return data_rows


def link_context_from_html_table(link, html_tables):
    """
    Extracts html_row_name and html_col_name if the link is embedded in a cell of an HTML table.
    Includes a reference to the parent HTML table.
    """
    html_row_name, html_col_name, html_table_reference = None, None, None
    link_cell = link.find_parent('td')  # Check if the link is in a table cell
    if link_cell:
        table = link_cell.find_parent('table')
        if table and table in html_tables:
            table_idx = html_tables.index(table)
            html_table_reference = f'HTML Table {table_idx}'
            row = link_cell.find_parent('tr')
            if row:
                row_idx = table.find_all('tr').index(row)
                col_idx = row.find_all(['td', 'th']).index(link_cell)
                # Extract row name and column name
                rows = table.find_all('tr')
                html_row_name = rows[row_idx].find('td').get_text(strip=True) if rows[row_idx].find('td') else None
                html_col_name = rows[0].find_all(['th', 'td'])[col_idx].get_text(strip=True) if rows[0].find_all(['th', 'td']) else None
    return {
        'html_row_name': html_row_name,
        'html_col_name': html_col_name,
        'html_table_reference': html_table_reference
    }


def extract_excel_file(link, base_url, html_context):
    excel_tables = []
    excel_url = urljoin(base_url, link['href'])
    try:
        response = requests.get(excel_url)
        response.raise_for_status()
        with open('temp.xlsx', 'wb') as temp_file:
            temp_file.write(response.content)

        xls = pd.ExcelFile('temp.xlsx')
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            context = {
                **html_context,
                'caption': f'Excel file: {sheet_name}',
                'file_name': link['href']
            }

            excel_tables.append({
                'context': context,
                'headers': [{'text': col} for col in df.columns],
                'data_rows': df.values.tolist(),
                'data_frame': df
            })
    except Exception as e:
        print(f"Failed to process Excel file {excel_url}: {e}")
    finally:
        if os.path.exists('temp.xlsx'):
            os.remove('temp.xlsx')
    return excel_tables


def extract_pdf_file(link, base_url, html_context):
    pdf_tables = []
    pdf_url = urljoin(base_url, link['href'])
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open('temp.pdf', 'wb') as temp_file:
            temp_file.write(response.content)

        context = {
            **html_context,
            'caption': 'PDF file',
            'file_name': link['href']
        }

        # Extract tables using tabula
        tables = read_pdf('temp.pdf', pages='all', multiple_tables=True, lattice=True)
        for idx, df in enumerate(tables):
            pdf_context = {**context, 'caption': f'PDF Table {idx + 1}'}
            pdf_tables.append({
                'context': pdf_context,
                'headers': [{'text': col} for col in df.columns],
                'data_rows': df.values.tolist(),
                'data_frame': df
            })

        # Extract additional text context from the PDF using PyPDF2
        pdf_text = extract_pdf_text('temp.pdf')
        context['surrounding_text'] += f" {pdf_text}"
    except Exception as e:
        print(f"Failed to process PDF file {pdf_url}: {e}")
    finally:
        if os.path.exists('temp.pdf'):
            os.remove('temp.pdf')
    return pdf_tables


def extract_pdf_text(pdf_path):
    text = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text.append(page.extract_text())
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
    return ' '.join(text)


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

    