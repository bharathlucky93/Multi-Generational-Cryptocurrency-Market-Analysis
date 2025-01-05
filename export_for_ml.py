import pandas as pd
from sqlalchemy import create_engine, text
import os

def connect_to_postgres():
    """Connect to PostgreSQL database"""
    try:
        engine = create_engine('postgresql+pg8000://postgres:11225@localhost:5432/crypto_data')
        print("Successfully connected to PostgreSQL")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        return None

def export_data_for_ml():
    """Export necessary data for machine learning analysis"""
    try:
        engine = connect_to_postgres()
        if engine is None:
            return

        # Create ml_data directory if it doesn't exist
        os.makedirs('ml_data', exist_ok=True)

        # Export price and volume data for time series analysis
        price_volume_query = """
            SELECT 
                b.time_period_start as timestamp,
                b.price_close as btc_price,
                b.volume_traded as btc_volume,
                e.price_close as eth_price,
                e.volume_traded as eth_volume,
                s.close as sol_price,
                s.volume as sol_volume
            FROM btc_data b
            JOIN eth_data e ON DATE(b.time_period_start) = DATE(e.time_period_start)
            JOIN sol_data s ON DATE(b.time_period_start) = DATE(s.date)
            ORDER BY b.time_period_start;
        """
        
        # Export daily returns for each cryptocurrency
        returns_query = """
            WITH daily_returns AS (
                SELECT 
                    DATE(time_period_start) as date,
                    (price_close - LAG(price_close) OVER (ORDER BY time_period_start)) / LAG(price_close) OVER (ORDER BY time_period_start) as btc_return,
                    (volume_traded - LAG(volume_traded) OVER (ORDER BY time_period_start)) / LAG(volume_traded) OVER (ORDER BY time_period_start) as btc_volume_change
                FROM btc_data
            ),
            eth_returns AS (
                SELECT 
                    DATE(time_period_start) as date,
                    (price_close - LAG(price_close) OVER (ORDER BY time_period_start)) / LAG(price_close) OVER (ORDER BY time_period_start) as eth_return,
                    (volume_traded - LAG(volume_traded) OVER (ORDER BY time_period_start)) / LAG(volume_traded) OVER (ORDER BY time_period_start) as eth_volume_change
                FROM eth_data
            ),
            sol_returns AS (
                SELECT 
                    DATE(date) as date,
                    (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) as sol_return,
                    (volume - LAG(volume) OVER (ORDER BY date)) / LAG(volume) OVER (ORDER BY date) as sol_volume_change
                FROM sol_data
            )
            SELECT 
                dr.date,
                dr.btc_return,
                dr.btc_volume_change,
                er.eth_return,
                er.eth_volume_change,
                sr.sol_return,
                sr.sol_volume_change
            FROM daily_returns dr
            JOIN eth_returns er ON dr.date = er.date
            JOIN sol_returns sr ON dr.date = sr.date
            ORDER BY dr.date;
        """

        # Export technical indicators
        technical_query = """
            WITH price_data AS (
                SELECT 
                    DATE(b.time_period_start) as date,
                    b.price_close as btc_price,
                    b.price_high as btc_high,
                    b.price_low as btc_low,
                    e.price_close as eth_price,
                    e.price_high as eth_high,
                    e.price_low as eth_low,
                    s.close as sol_price,
                    s.high as sol_high,
                    s.low as sol_low
                FROM btc_data b
                JOIN eth_data e ON DATE(b.time_period_start) = DATE(e.time_period_start)
                JOIN sol_data s ON DATE(b.time_period_start) = DATE(s.date)
                ORDER BY date
            )
            SELECT *,
                AVG(btc_price) OVER (ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as btc_sma_14,
                AVG(eth_price) OVER (ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as eth_sma_14,
                AVG(sol_price) OVER (ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as sol_sma_14,
                AVG(btc_price) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as btc_sma_30,
                AVG(eth_price) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as eth_sma_30,
                AVG(sol_price) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as sol_sma_30
            FROM price_data;
        """

        # Execute queries and save to CSV
        print("Exporting price and volume data...")
        price_volume_df = pd.read_sql(price_volume_query, engine)
        price_volume_df.to_csv('ml_data/price_volume_data.csv', index=False)
        print(f"Exported {len(price_volume_df)} records to price_volume_data.csv")

        print("Exporting returns data...")
        returns_df = pd.read_sql(returns_query, engine)
        returns_df.to_csv('ml_data/returns_data.csv', index=False)
        print(f"Exported {len(returns_df)} records to returns_data.csv")

        print("Exporting technical indicators...")
        technical_df = pd.read_sql(technical_query, engine)
        technical_df.to_csv('ml_data/technical_data.csv', index=False)
        print(f"Exported {len(technical_df)} records to technical_data.csv")

        print("\nAll data exported successfully to ml_data directory")
        print("\nFiles created:")
        print("1. ml_data/price_volume_data.csv - For time series forecasting")
        print("2. ml_data/returns_data.csv - For pattern recognition")
        print("3. ml_data/technical_data.csv - For technical analysis")

    except Exception as e:
        print(f"Error exporting data: {str(e)}")

if __name__ == "__main__":
    export_data_for_ml() 