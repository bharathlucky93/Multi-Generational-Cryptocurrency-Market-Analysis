import yfinance as yf
import pandas as pd
from datetime import datetime

def get_solana_data():
    # Define the date range (extended to match other data)
    start_date = "2021-01-01"
    end_date = "2024-08-31"
    
    print(f"Fetching Solana data from {start_date} to {end_date}...")
    
    try:
        # Create Ticker object for Solana
        sol = yf.Ticker("SOL-USD")
        
        # Download historical data
        df = sol.history(start=start_date, end=end_date, interval="1d")
        
        # Save to CSV
        filename = f'SOLANA_SOL_USD_{start_date}_{end_date}.csv'
        df.to_csv(filename)
        
        print(f"\nData saved to {filename}")
        print("\nFirst few rows of data:")
        print(df.head())
        
        print("\nSummary statistics:")
        print(df.describe())
        
        print(f"\nTotal records: {len(df)}")
        if len(df) < 1000:
            print(f"Warning: Only {len(df)} records found. Consider adjusting date range.")
        
        return df
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting Solana data collection...")
    df = get_solana_data() 