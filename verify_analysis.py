import pandas as pd
from sqlalchemy import create_engine, text

def connect_to_postgres():
    """Connect to PostgreSQL database"""
    try:
        engine = create_engine('postgresql+pg8000://postgres:11225@localhost:5432/crypto_data')
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        return None

def print_analysis_results(engine):
    """Print analysis results from PostgreSQL"""
    try:
        # Print basic statistics
        print("\nBASIC STATISTICS:")
        stats_query = """
            SELECT crypto, metric, value
            FROM analysis_results
            ORDER BY crypto, metric;
        """
        stats_df = pd.read_sql(stats_query, engine)
        
        for crypto in stats_df['crypto'].unique():
            print(f"\n{crypto} Statistics:")
            crypto_stats = stats_df[stats_df['crypto'] == crypto]
            for _, row in crypto_stats.iterrows():
                print(f"  {row['metric']}: {row['value']:.6f}")

        # Print correlations
        print("\nCORRELATIONS:")
        corr_query = """
            SELECT correlation_type, crypto1, crypto2, correlation_value
            FROM correlations
            ORDER BY correlation_type, crypto1, crypto2;
        """
        corr_df = pd.read_sql(corr_query, engine)
        
        for corr_type in corr_df['correlation_type'].unique():
            print(f"\n{corr_type.upper()} Correlations:")
            type_corr = corr_df[corr_df['correlation_type'] == corr_type]
            for _, row in type_corr.iterrows():
                print(f"  {row['crypto1']} vs {row['crypto2']}: {row['correlation_value']:.6f}")

    except Exception as e:
        print(f"Error printing analysis results: {str(e)}")

def main():
    try:
        # Connect to PostgreSQL
        engine = connect_to_postgres()
        if engine is None:
            return

        # Print analysis results
        print_analysis_results(engine)

    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 