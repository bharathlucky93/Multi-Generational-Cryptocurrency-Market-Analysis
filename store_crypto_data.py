import pandas as pd
import json
from pymongo import MongoClient
from datetime import datetime
from decimal import Decimal

def connect_to_mongodb():
    """Connect to MongoDB and return database instance"""
    try:
        # Connect to MongoDB (running on default port 27017)
        client = MongoClient('mongodb://localhost:27017/')
        
        # Test the connection
        client.server_info()
        
        # Create/Get database
        db = client['CrytoAnalysis']
        print("Successfully connected to MongoDB")
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        return None

def load_json_data(file_path):
    """Load data from JSON file"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        return None

def process_document(doc):
    """Process document to handle large numbers"""
    if isinstance(doc, dict):
        return {k: process_document(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [process_document(v) for v in doc]
    elif isinstance(doc, (int, float)):
        # Convert to string representation to avoid precision issues
        return str(doc)
    return doc

def store_crypto_data(db):
    """Store cryptocurrency data in MongoDB"""
    try:
        # Drop all existing collections
        collections_to_drop = ['symbols', 'BTC', 'ETH', 'SOL']
        for collection_name in collections_to_drop:
            db[collection_name].drop()
            print(f"Dropped existing {collection_name} collection")

        # Store symbols data first
        symbols_data = load_json_data('symbols_data_20241229_124605.json')
        if symbols_data:
            # Process and store symbols data
            processed_symbols = [process_document(doc) for doc in symbols_data]
            symbols_collection = db['symbols']
            symbols_collection.insert_many(processed_symbols)
            print(f"Successfully stored {len(processed_symbols)} symbol records")

        # Store BTC data
        btc_data = load_json_data('BITSTAMP_SPOT_BTC_USD_20210101_20240831.json')
        if btc_data:
            processed_btc = [process_document(doc) for doc in btc_data]
            btc_collection = db['BTC']
            btc_collection.insert_many(processed_btc)
            print(f"Successfully stored {len(processed_btc)} BTC records")

        # Store ETH data
        eth_data = load_json_data('BITSTAMP_SPOT_ETH_USD_20210101_20240831.json')
        if eth_data:
            processed_eth = [process_document(doc) for doc in eth_data]
            eth_collection = db['ETH']
            eth_collection.insert_many(processed_eth)
            print(f"Successfully stored {len(processed_eth)} ETH records")

        # Store SOL data from CSV
        sol_df = pd.read_csv('SOLANA_SOL_USD_2021-01-01_2024-08-31.csv')
        
        # Convert DataFrame to list of dictionaries and process date
        sol_records = []
        for index, row in sol_df.iterrows():
            record = row.to_dict()
            # Convert the index (date) to proper format
            record['time_period_start'] = str(index)  # Convert to string to match format
            record['time_period_end'] = str(index)
            record['time_open'] = str(index)
            record['time_close'] = str(index)
            # Process numeric values
            sol_records.append(process_document(record))

        if sol_records:
            sol_collection = db['SOL']
            sol_collection.insert_many(sol_records)
            print(f"Successfully stored {len(sol_records)} SOL records")

    except Exception as e:
        print(f"Error storing data: {str(e)}")

def main():
    # Connect to MongoDB
    db = connect_to_mongodb()
    if db is not None:
        # Store all cryptocurrency data
        store_crypto_data(db)
        print("Data storage process completed")

if __name__ == "__main__":
    main() 