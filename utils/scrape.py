import requests


def process_response(self, response, options):

    url_hash = hashlib.md5(response.url.encode()).hexdigest()
    soup = BeautifulSoup(response.text, 'html.parser')

    all_tables = []

    if options['data_format'] == "HTML Tables":
        tables = soup.find_all('table')
        for table in tables:
            df = pd.read_html(str(table))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns.values]
            all_tables.append(df)
    elif options['data_format'] == "Excel Files":
        links = [link['href'] for link in soup.find_all('a', href=True) if link['href'].endswith(('.xls', '.xlsx'))]
        for link in links:
            excel_url = requests.compat.urljoin(response.url, link)
            try:
                excel_response = requests.get(excel_url)
                excel_response.raise_for_status()
                df = pd.read_excel(BytesIO(excel_response.content))
                all_tables.append(df)
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Failed to download the Excel file: {e}")
    if all_tables:
        self.save_processed_data(url_hash, response.url, all_tables)

    return {
        "source": response.url,
        "tables": all_tables
    }


def scrape_tables(url, options):

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None

    tables = process_response(response, options)

    return tables