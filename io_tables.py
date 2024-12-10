import os
import pickle
import hashlib

# I/O FUNCTIONS

def load_scraped_data():
    """Load all pickles from data directory. Return empty dict if no data found."""

    data_dict = {}
    if os.path.exists("data"):
        files = [file for file in os.listdir("data") if file.endswith('.pkl')]
        for file in files:
            with open(f"data/{file}", "rb") as f:
                data = pickle.load(f)
                data_dict[data['source']] = data
    return data_dict

def save_harvested_tables(source_data):
    """Create a dict for data store and pickle it."""

    if not os.path.exists("data"):
        os.makedirs("data")

    # Save scraped source data in a file named using hash
    url_hash = hashlib.md5(source_data['source'].encode()).hexdigest()

    with open(f"data/{url_hash}.pkl", "wb") as f:
        pickle.dump(source_data, f)