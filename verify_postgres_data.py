from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime

def format_record(record, table_name):
    """Format a database record for display"""
    try:
        if table_name in ['btc_data', 'eth_data']:
            return {
                'time_period_start': record[1].strftime('%Y-%m-%d %H:%M:%S'),
                'time_period_end': record[2].strftime('%Y-%m-%d %H:%M:%S'),
                'price_open': f"${float(record[5]):,.2f}",
                'price_high': f"${float(record[6]):,.2f}",
                'price_low': f"${float(record[7]):,.2f}",
                'price_close': f"${float(record[8]):,.2f}",
                'volume_traded': f"{float(record[9]):,.2f}",
                'trades_count': f"{int(record[10]):,}"
            }
        else:  # sol_data
            return {
                'date': record[1].strftime('%Y-%m-%d %H:%M:%S'),
                'open': f"${float(record[2]):,.2f}",
                'high': f"${float(record[3]):,.2f}",
                'low': f"${float(record[4]):,.2f}",
                'close': f"${float(record[5]):,.2f}",
                'volume': f"{float(record[6]):,.2f}",
                'dividends': f"{float(record[7]):,.2f}",
                'stock_splits': f"{float(record[8]):,.2f}"
            }
    except Exception as e:
        print(f"Error formatting record: {str(e)}")
        print(f"Raw record: {record}")
        return None

def format_date(dt):
    """Format datetime object to string"""
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    return str(dt)

def get_table_stats(conn, table_name):
    """Get basic statistics for a table"""
    try:
        # Get record count
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        
        # Get sample record
        sample = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1")).fetchone()
        formatted_sample = format_record(sample, table_name) if sample else None
        
        # Get date range
        if table_name in ['btc_data', 'eth_data']:
            date_range = conn.execute(text(f"""
                SELECT MIN(time_period_start), MAX(time_period_start)
                FROM {table_name}
            """)).fetchone()
        else:
            date_range = conn.execute(text(f"""
                SELECT MIN(date), MAX(date)
                FROM {table_name}
            """)).fetchone()
        
        # Get price range
        if table_name in ['btc_data', 'eth_data']:
            price_range = conn.execute(text(f"""
                SELECT MIN(price_close), MAX(price_close), AVG(price_close)
                FROM {table_name}
            """)).fetchone()
        else:
            price_range = conn.execute(text(f"""
                SELECT MIN(close), MAX(close), AVG(close)
                FROM {table_name}
            """)).fetchone()
        
        return {
            'count': count,
            'sample': formatted_sample,
            'date_range': (format_date(date_range[0]), format_date(date_range[1])) if date_range else None,
            'price_range': {
                'min': f"${float(price_range[0]):,.2f}",
                'max': f"${float(price_range[1]):,.2f}",
                'avg': f"${float(price_range[2]):,.2f}"
            } if price_range else None
        }
    except Exception as e:
        print(f"Error getting stats for {table_name}: {str(e)}")
        return None

def verify_data():
    """Verify data stored in PostgreSQL"""
    try:
        # Connect to PostgreSQL
        engine = create_engine('postgresql+pg8000://postgres:11225@localhost:5432/crypto_data')
        print("Successfully connected to PostgreSQL")

        with engine.connect() as conn:
            tables = ['btc_data', 'eth_data', 'sol_data']
            
            for table in tables:
                print(f"\n{table.upper().replace('_', ' ')}:")
                stats = get_table_stats(conn, table)
                
                if stats:
                    print(f"Total records: {stats['count']:,}")
                    if stats['date_range']:
                        print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
                    if stats['price_range']:
                        print(f"Price range:")
                        print(f"  Min: {stats['price_range']['min']}")
                        print(f"  Max: {stats['price_range']['max']}")
                        print(f"  Avg: {stats['price_range']['avg']}")
                    if stats['sample']:
                        print("\nSample record:")
                        for key, value in stats['sample'].items():
                            print(f"  {key}: {value}")

            # Print summary
            print("\nSUMMARY:")
            total_records = 0
            for table in tables:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                total_records += count
                print(f"{table.upper().replace('_', ' ')}: {count:,} records")
            print(f"Total records across all tables: {total_records:,}")

        print("\nVerification completed")

    except Exception as e:
        print(f"Error verifying data: {str(e)}")

if __name__ == "__main__":
    verify_data() 