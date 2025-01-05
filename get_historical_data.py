import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import os
import time

def get_historical_data_chunk(symbol, start_date, end_date, period_id="1DAY"):
    """Get historical data for a specific time chunk"""
    # API endpoint
    base_url = "https://rest.coinapi.io/v1/ohlcv"
    
    # Headers with API key
    headers = {
        'Accept': 'application/json',
        'X-CoinAPI-Key': '3683B432-DC78-4C48-8B12-A8BD5A8FC2D4'
    }
    
    # Construct the full URL
    url = f"{base_url}/{symbol}/history"
    
    # Parameters for the request
    params = {
        'period_id': period_id,
        'time_start': start_date.isoformat(),
        'time_end': end_date.isoformat(),
        'limit': 100000
    }
    
    try:
        print(f"Making request for period {start_date.date()} to {end_date.date()}")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_historical_data(symbol, start_date, end_date, period_id="1DAY"):
    """Get all historical data by making multiple requests"""
    all_data = []
    current_start = start_date
    
    # Get data in 6-month chunks to ensure we get all records
    chunk_size = timedelta(days=180)
    
    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        chunk_data = get_historical_data_chunk(symbol, current_start, current_end, period_id)
        
        if chunk_data:
            all_data.extend(chunk_data)
            print(f"Retrieved {len(chunk_data)} records for period {current_start.date()} to {current_end.date()}")
        else:
            print(f"Failed to retrieve data for period {current_start.date()} to {current_end.date()}")
        
        current_start = current_end
        time.sleep(1)  # Rate limiting
    
    if all_data:
        print(f"\nTotal records retrieved: {len(all_data)}")
        
        # Convert to DataFrame for statistics
        df = pd.DataFrame(all_data)
        
        # Save to JSON
        filename = f'{symbol}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.json'
        with open(filename, 'w') as f:
            json.dump(all_data, f, indent=4)
        
        file_size = os.path.getsize(filename)
        print(f"Data saved to {filename} (Size: {file_size} bytes)")
        
        print("\nFirst few rows of data:")
        print(df.head())
        
        return df
    
    return None

def main():
    # Define date range - extended to get more data
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 8, 31)
    
    # Symbols for BTC and ETH (SPOT prices from Bitstamp)
    symbols = [
        "BITSTAMP_SPOT_BTC_USD",
        "BITSTAMP_SPOT_ETH_USD"
    ]
    
    # Fetch data for each symbol
    for symbol in symbols:
        print(f"\nFetching historical data for {symbol}...")
        df = get_historical_data(symbol, start_date, end_date)
        
        if df is not None:
            print(f"\nSummary statistics for {symbol}:")
            print(df.describe())
            print(f"Total records: {len(df)}")
            if len(df) < 1000:
                print(f"Warning: Only {len(df)} records found. Consider adjusting date range.")
        
        # Add a delay between symbols to avoid rate limiting
        time.sleep(2)

if __name__ == "__main__":
    print("Starting historical data collection...")
    main() 