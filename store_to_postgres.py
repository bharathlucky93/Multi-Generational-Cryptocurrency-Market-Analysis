import pandas as pd
import json
from datetime import datetime
from sqlalchemy import create_engine, text

def connect_to_postgres():
    """Connect to PostgreSQL database"""
    try:
        engine = create_engine('postgresql+pg8000://postgres:11225@localhost:5432/crypto_data')
        print("Successfully connected to PostgreSQL")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        return None

def create_tables(engine):
    """Create tables for cryptocurrency data"""
    try:
        # Create tables for each cryptocurrency
        create_table_queries = [
            """
            DROP TABLE IF EXISTS btc_data;
            """,
            """
            CREATE TABLE btc_data (
                id SERIAL PRIMARY KEY,
                time_period_start TIMESTAMP,
                time_period_end TIMESTAMP,
                time_open TIMESTAMP,
                time_close TIMESTAMP,
                price_open DECIMAL(18,8),
                price_high DECIMAL(18,8),
                price_low DECIMAL(18,8),
                price_close DECIMAL(18,8),
                volume_traded DECIMAL(18,8),
                trades_count INTEGER
            );
            """,
            """
            DROP TABLE IF EXISTS eth_data;
            """,
            """
            CREATE TABLE eth_data (
                id SERIAL PRIMARY KEY,
                time_period_start TIMESTAMP,
                time_period_end TIMESTAMP,
                time_open TIMESTAMP,
                time_close TIMESTAMP,
                price_open DECIMAL(18,8),
                price_high DECIMAL(18,8),
                price_low DECIMAL(18,8),
                price_close DECIMAL(18,8),
                volume_traded DECIMAL(18,8),
                trades_count INTEGER
            );
            """,
            """
            DROP TABLE IF EXISTS sol_data;
            """,
            """
            CREATE TABLE sol_data (
                id SERIAL PRIMARY KEY,
                date TIMESTAMP,
                open DECIMAL(18,8),
                high DECIMAL(18,8),
                low DECIMAL(18,8),
                close DECIMAL(18,8),
                volume DECIMAL(30,8),
                dividends DECIMAL(18,8),
                stock_splits DECIMAL(18,8)
            );
            """
        ]
        
        with engine.connect() as conn:
            for query in create_table_queries:
                conn.execute(text(query))
                conn.commit()
                print("Executed table creation/drop query")
        
        print("Successfully created tables")
        
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        raise

def load_json_data(file_path):
    """Load data from JSON file"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        return None

def process_crypto_data(data, crypto_type):
    """Process cryptocurrency data into a DataFrame with proper types"""
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime objects
    timestamp_columns = ['time_period_start', 'time_period_end', 'time_open', 'time_close']
    for col in timestamp_columns:
        df[col] = pd.to_datetime(df[col])
    
    # Convert numeric columns to appropriate types
    df['price_open'] = pd.to_numeric(df['price_open'])
    df['price_high'] = pd.to_numeric(df['price_high'])
    df['price_low'] = pd.to_numeric(df['price_low'])
    df['price_close'] = pd.to_numeric(df['price_close'])
    df['volume_traded'] = pd.to_numeric(df['volume_traded'])
    df['trades_count'] = pd.to_numeric(df['trades_count'], downcast='integer')
    
    return df

def insert_sol_data(engine, sol_df):
    """Insert Solana data using direct SQL commands"""
    try:
        with engine.connect() as conn:
            for index, row in sol_df.iterrows():
                insert_query = text("""
                    INSERT INTO sol_data (date, open, high, low, close, volume, dividends, stock_splits)
                    VALUES (:date, :open, :high, :low, :close, :volume, :dividends, :stock_splits)
                """)
                
                conn.execute(insert_query, {
                    'date': pd.to_datetime(row['Date']),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'dividends': float(row['Dividends']),
                    'stock_splits': float(row['Stock Splits'])
                })
                
                if index % 100 == 0:
                    print(f"Inserted {index} SOL records...")
                
            conn.commit()
            print(f"Successfully inserted {len(sol_df)} SOL records")
    except Exception as e:
        print(f"Error inserting SOL data: {str(e)}")
        raise

def insert_dataframe_in_chunks(df, table_name, engine, chunk_size=100):
    """Insert DataFrame into PostgreSQL in chunks"""
    total_rows = len(df)
    chunks = range(0, total_rows, chunk_size)
    
    for i in chunks:
        chunk_df = df.iloc[i:i + chunk_size]
        try:
            chunk_df.to_sql(table_name, engine, if_exists='append', index=False)
            print(f"Inserted chunk {i//chunk_size + 1} of {len(chunks)} into {table_name}")
        except Exception as e:
            print(f"Error inserting chunk into {table_name}: {str(e)}")
            raise

def insert_crypto_data(engine):
    """Insert cryptocurrency data into PostgreSQL"""
    try:
        # Insert BTC data
        btc_data = load_json_data('BITSTAMP_SPOT_BTC_USD_20210101_20240831.json')
        if btc_data:
            print(f"Loaded {len(btc_data)} BTC records from JSON")
            btc_df = process_crypto_data(btc_data, 'BTC')
            insert_dataframe_in_chunks(btc_df, 'btc_data', engine)
            print(f"Successfully inserted {len(btc_data)} BTC records")
        
        # Insert ETH data
        eth_data = load_json_data('BITSTAMP_SPOT_ETH_USD_20210101_20240831.json')
        if eth_data:
            print(f"Loaded {len(eth_data)} ETH records from JSON")
            eth_df = process_crypto_data(eth_data, 'ETH')
            insert_dataframe_in_chunks(eth_df, 'eth_data', engine)
            print(f"Successfully inserted {len(eth_data)} ETH records")
        
        # Insert SOL data
        sol_df = pd.read_csv('SOLANA_SOL_USD_2021-01-01_2024-08-31.csv')
        if not sol_df.empty:
            print(f"Loaded {len(sol_df)} SOL records from CSV")
            insert_sol_data(engine, sol_df)
        
        print("All data successfully inserted")
        
    except Exception as e:
        print(f"Error inserting data: {str(e)}")
        raise

def main():
    try:
        # Connect to PostgreSQL
        engine = connect_to_postgres()
        if engine is not None:
            # Create tables
            create_tables(engine)
            
            # Insert data
            insert_crypto_data(engine)
            
            print("Data storage process completed")
    except Exception as e:
        print(f"Main error: {str(e)}")

if __name__ == "__main__":
    main() 