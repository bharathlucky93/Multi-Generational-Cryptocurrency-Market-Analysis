import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from datetime import datetime
import sys
import os
import json
import traceback

def connect_to_postgres():
    """Connect to PostgreSQL database"""
    try:
        print("Attempting to connect to PostgreSQL...")
        engine = create_engine('postgresql+pg8000://postgres:11225@localhost:5432/crypto_data')
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Successfully connected to PostgreSQL")
        return engine
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        return None

def fetch_crypto_data(engine):
    """Fetch cryptocurrency data from PostgreSQL with improved error handling"""
    try:
        # Fetch BTC data
        btc_query = """
            SELECT time_period_start::timestamp as date,
                   price_close as btc_price,
                   volume_traded as btc_volume
            FROM btc_data 
            ORDER BY time_period_start;
        """
        btc_df = pd.read_sql(btc_query, engine, parse_dates=['date'])
        btc_df.set_index('date', inplace=True)
        print(f"Retrieved {len(btc_df)} BTC records")
        
        # Fetch ETH data
        print("Fetching ETH data...")
        eth_query = """
            SELECT time_period_start::timestamp as date,
                   price_close as eth_price,
                   volume_traded as eth_volume
            FROM eth_data 
            ORDER BY time_period_start;
        """
        eth_df = pd.read_sql(eth_query, engine, parse_dates=['date'])
        eth_df.set_index('date', inplace=True)
        print(f"Retrieved {len(eth_df)} ETH records")
        
        # Fetch SOL data
        print("Fetching SOL data...")
        sol_query = """
            SELECT date::timestamp as date,
                   close as sol_price,
                   volume as sol_volume
            FROM sol_data 
            ORDER BY date;
        """
        sol_df = pd.read_sql(sol_query, engine, parse_dates=['date'])
        sol_df.set_index('date', inplace=True)
        print(f"Retrieved {len(sol_df)} SOL records")
        
        # Create a common date range
        start_date = min(btc_df.index.min(), eth_df.index.min(), sol_df.index.min())
        end_date = max(btc_df.index.max(), eth_df.index.max(), sol_df.index.max())
        all_dates = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        # Convert index to date for consistent merging
        btc_df.index = btc_df.index.date
        eth_df.index = eth_df.index.date
        sol_df.index = sol_df.index.date
        
        # Reindex all dataframes to common date range and forward fill missing values
        btc_df = btc_df.reindex(all_dates).ffill()
        eth_df = eth_df.reindex(all_dates).ffill()
        sol_df = sol_df.reindex(all_dates).ffill()
        
        # Merge dataframes on date
        df = pd.concat([btc_df, eth_df, sol_df], axis=1)
        
        # Verify data is loaded correctly
        print(f"\nCombined dataframe shape: {df.shape}")
        print("Available columns:", df.columns.tolist())
        
        for col in ['btc_price', 'eth_price', 'sol_price']:
            if col not in df.columns:
                print(f"Error: {col} not found in dataframe")
                print(f"Available columns: {df.columns.tolist()}")
                return None
            if df[col].isnull().all():
                print(f"Error: {col} contains all null values")
                return None
            print(f"\nColumn {col}:")
            print(f"Range: {df[col].min():.2f} to {df[col].max():.2f}")
            print(f"Null values: {df[col].isnull().sum()}")
            print(f"First 5 values:\n{df[col].head()}")
            print(f"Last 5 values:\n{df[col].tail()}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")
        traceback.print_exc()
        return None

def calculate_basic_stats(df):
    """Calculate basic statistical metrics"""
    try:
        print("Calculating basic statistics...")
        stats = {}
        
        # Calculate daily returns
        price_cols = ['btc_price', 'eth_price', 'sol_price']
        for col in price_cols:
            df[f'{col}_return'] = df[col].pct_change()
            print(f"Calculated returns for {col}")

        # Calculate volatility metrics
        for col in price_cols:
            returns_col = f'{col}_return'
            stats[col] = {
                'daily_volatility': df[returns_col].std(),
                'annualized_volatility': df[returns_col].std() * np.sqrt(252),
                'variance': df[returns_col].var(),
                'mean_return': df[returns_col].mean(),
                'median_return': df[returns_col].median(),
                'quartiles': df[returns_col].quantile([0.25, 0.75]).to_dict(),
                'skewness': df[returns_col].skew(),
                'kurtosis': df[returns_col].kurtosis()
            }
            print(f"Calculated volatility metrics for {col}")

        # Calculate moving averages
        for col in price_cols:
            df[f'{col}_MA7'] = df[col].rolling(window=7).mean()
            df[f'{col}_MA30'] = df[col].rolling(window=30).mean()
            print(f"Calculated moving averages for {col}")

        return df, stats

    except Exception as e:
        print(f"Error calculating basic stats: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")
        return None, None

def calculate_correlations(df):
    """Calculate correlation metrics"""
    try:
        print("Calculating correlations...")
        # Price correlations
        price_cols = ['btc_price', 'eth_price', 'sol_price']
        price_corr = df[price_cols].corr()
        print("Calculated price correlations")

        # Volume correlations
        volume_cols = ['btc_volume', 'eth_volume', 'sol_volume']
        volume_corr = df[volume_cols].corr()
        print("Calculated volume correlations")

        # Returns correlations
        return_cols = ['btc_price_return', 'eth_price_return', 'sol_price_return']
        return_corr = df[return_cols].corr()
        print("Calculated returns correlations")

        return price_corr, volume_corr, return_corr

    except Exception as e:
        print(f"Error calculating correlations: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")
        return None, None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    try:
        print("Calculating technical indicators...")
        price_cols = ['btc_price', 'eth_price', 'sol_price']
        
        for col in price_cols:
            # RSI (14-day)
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df[f'{col}_RSI'] = 100 - (100 / (1 + rs))
            print(f"Calculated RSI for {col}")

            # MACD
            exp1 = df[col].ewm(span=12, adjust=False).mean()
            exp2 = df[col].ewm(span=26, adjust=False).mean()
            df[f'{col}_MACD'] = exp1 - exp2
            df[f'{col}_Signal'] = df[f'{col}_MACD'].ewm(span=9, adjust=False).mean()
            print(f"Calculated MACD for {col}")

            # Bollinger Bands
            df[f'{col}_MA20'] = df[col].rolling(window=20).mean()
            df[f'{col}_BB_upper'] = df[f'{col}_MA20'] + 2 * df[col].rolling(window=20).std()
            df[f'{col}_BB_lower'] = df[f'{col}_MA20'] - 2 * df[col].rolling(window=20).std()
            print(f"Calculated Bollinger Bands for {col}")

        return df

    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")
        return None

def store_analysis_results(engine, df, stats, price_corr, volume_corr, return_corr):
    """Store analysis results in PostgreSQL"""
    try:
        print("Creating tables for analysis results...")
        # Create tables for analysis results
        create_tables_query = """
        DROP TABLE IF EXISTS analysis_results;
        CREATE TABLE analysis_results (
            crypto VARCHAR(10),
            metric VARCHAR(50),
            value FLOAT
        );

        DROP TABLE IF EXISTS correlations;
        CREATE TABLE correlations (
            correlation_type VARCHAR(20),
            crypto1 VARCHAR(10),
            crypto2 VARCHAR(10),
            correlation_value FLOAT
        );
        """
        
        with engine.connect() as conn:
            conn.execute(text(create_tables_query))
            conn.commit()
        print("Created analysis results tables")

        # Store basic stats
        print("Storing basic statistics...")
        stats_records = []
        for crypto, metrics in stats.items():
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for q, v in value.items():
                        stats_records.append({
                            'crypto': crypto.split('_')[0].upper(),
                            'metric': f"{metric}_{q}",
                            'value': v
                        })
                else:
                    stats_records.append({
                        'crypto': crypto.split('_')[0].upper(),
                        'metric': metric,
                        'value': value
                    })
        
        stats_df = pd.DataFrame(stats_records)
        stats_df.to_sql('analysis_results', engine, if_exists='append', index=False)
        print(f"Stored {len(stats_records)} statistical records")

        # Store correlations
        print("Storing correlations...")
        correlation_records = []
        for corr_type, corr_df in [('price', price_corr), ('volume', volume_corr), ('return', return_corr)]:
            for i in corr_df.index:
                for j in corr_df.columns:
                    if i != j:
                        correlation_records.append({
                            'correlation_type': corr_type,
                            'crypto1': i.split('_')[0].upper(),
                            'crypto2': j.split('_')[0].upper(),
                            'correlation_value': corr_df.loc[i, j]
                        })
        
        corr_df = pd.DataFrame(correlation_records)
        corr_df.to_sql('correlations', engine, if_exists='append', index=False)
        print(f"Stored {len(correlation_records)} correlation records")

        print("Analysis results successfully stored in PostgreSQL")

    except Exception as e:
        print(f"Error storing analysis results: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")

def create_price_plot(df):
    """Create improved price comparison plot with better visibility"""
    try:
        plt.style.use('default')
        sns.set_style("whitegrid")
        
        # Verify data before plotting
        print("\nVerifying data before plotting:")
        for col in ['btc_price', 'eth_price', 'sol_price']:
            print(f"{col} - Records: {len(df[col])}, Range: {df[col].min():.2f} to {df[col].max():.2f}")
            print(f"{col} - First 5 values: {df[col].head()}")
            print(f"{col} - Last 5 values: {df[col].tail()}")
        
        # Create figure with 4 subplots
        fig = plt.figure(figsize=(15, 20))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 1.5, 1], hspace=0.4)
        
        # Individual price plots for each cryptocurrency
        cryptos = [('btc_price', 'BTC', 'blue'), ('eth_price', 'ETH', 'orange'), ('sol_price', 'SOL', 'green')]
        
        for idx, (col, label, color) in enumerate(cryptos):
            ax = fig.add_subplot(gs[idx])
            
            # Ensure data is sorted by date
            plot_data = df.sort_index()
            
            # Plot price line with increased line width
            price_line = ax.plot(plot_data.index, plot_data[col], label=f'{label} Price', color=color, linewidth=2.5)
            
            # Add price statistics
            max_price = plot_data[col].max()
            min_price = plot_data[col].min()
            current_price = plot_data[col].iloc[-1]
            pct_change = ((current_price - plot_data[col].iloc[0]) / plot_data[col].iloc[0]) * 100
            
            # Add 50-day moving average
            ma50 = plot_data[col].rolling(window=50, min_periods=1).mean()
            ma_line = ax.plot(plot_data.index, ma50, label='50-day MA', color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Set appropriate y-axis limits with padding
            y_min = plot_data[col].min() * 0.95
            y_max = plot_data[col].max() * 1.05
            ax.set_ylim(y_min, y_max)
            
            # Add annotations with improved formatting
            ax.annotate(
                f'{label} Statistics:\nCurrent: ${current_price:,.2f}\nMax: ${max_price:,.2f}\nMin: ${min_price:,.2f}\n'
                f'Total Return: {pct_change:,.1f}%\nMA(50): ${ma50.iloc[-1]:,.2f}',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'),
                verticalalignment='top'
            )
            
            # Improve axis labels and formatting
            ax.set_title(f'{label} Price History', fontsize=13, pad=15, weight='bold')
            ax.set_ylabel('Price (USD)', fontsize=11, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax.tick_params(axis='both', labelsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Add minor gridlines for better readability
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            
            # Set number of minor ticks
            ax.yaxis.set_minor_locator(plt.MultipleLocator(base=(y_max - y_min)/40))
        
        # Percentage change comparison plot
        ax_pct = fig.add_subplot(gs[3])
        for col, label, color in cryptos:
            normalized = (plot_data[col] / plot_data[col].iloc[0] - 1) * 100
            ax_pct.plot(plot_data.index, normalized, label=f'{label}', color=color, linewidth=2.5)
        
        # Improve percentage change plot formatting
        ax_pct.set_title('Percentage Change from Initial Price', fontsize=13, pad=15, weight='bold')
        ax_pct.set_xlabel('Date', fontsize=11, weight='bold')
        ax_pct.set_ylabel('% Change', fontsize=11, weight='bold')
        ax_pct.legend(fontsize=10, framealpha=0.9)
        ax_pct.grid(True, linestyle='--', alpha=0.7)
        ax_pct.grid(True, which='minor', linestyle=':', alpha=0.4)
        
        # Set number of minor ticks for percentage plot
        y_min, y_max = ax_pct.get_ylim()
        ax_pct.yaxis.set_minor_locator(plt.MultipleLocator(base=(y_max - y_min)/40))
        
        ax_pct.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}%'))
        ax_pct.tick_params(axis='both', labelsize=10)
        ax_pct.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Cryptocurrency Price Analysis (2021-2024)', fontsize=16, y=0.95, weight='bold')
        plt.tight_layout()
        plt.savefig('analysis_results/plots/price_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nPrice plot created successfully")
        
    except Exception as e:
        print(f"\nError creating price plot: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")
        traceback.print_exc()

def create_volume_plot(df):
    """Create improved volume comparison plot with better visibility"""
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 1.5, 1], hspace=0.3)
    
    cryptos = [('btc_volume', 'BTC', 'blue'), ('eth_volume', 'ETH', 'orange'), ('sol_volume', 'SOL', 'green')]
    
    for idx, (col, label, color) in enumerate(cryptos):
        ax = fig.add_subplot(gs[idx])
        
        # Create bar plot for volume
        volume = df[col]
        ax.bar(df.index, volume, alpha=0.3, color=color, label=f'{label} Volume')
        
        # Add 20-day moving average
        ma20 = volume.rolling(window=20).mean()
        ax.plot(df.index, ma20, color='red', linewidth=2, label='20-day MA', alpha=0.8)
        
        # Calculate and add volume statistics
        avg_vol = volume.mean()
        max_vol = volume.max()
        current_vol = volume.iloc[-1]
        vol_change = ((current_vol - volume.iloc[0]) / volume.iloc[0]) * 100
        
        # Add annotations
        ax.annotate(
            f'{label} Volume Statistics:\nCurrent: {current_vol:,.0f}\nAverage: {avg_vol:,.0f}\n'
            f'Maximum: {max_vol:,.0f}\nVolume Change: {vol_change:,.1f}%',
            xy=(0.02, 0.95), xycoords='axes fraction',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            verticalalignment='top'
        )
        
        ax.set_title(f'{label} Trading Volume', fontsize=12, pad=10)
        ax.set_ylabel('Volume', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.tick_params(axis='x', rotation=45)
    
    # Volume comparison as percentage of total
    ax_pct = fig.add_subplot(gs[3])
    total_volume = df[[col for col, _, _ in cryptos]].sum(axis=1)
    
    bottom = np.zeros(len(df))
    for col, label, color in cryptos:
        pct = (df[col] / total_volume) * 100
        ax_pct.bar(df.index, pct, bottom=bottom, label=f'{label}', alpha=0.7, color=color)
        bottom += pct
    
    ax_pct.set_title('Volume Distribution (% of Total)', fontsize=12)
    ax_pct.set_xlabel('Date', fontsize=10)
    ax_pct.set_ylabel('% of Total Volume', fontsize=10)
    ax_pct.legend()
    ax_pct.grid(True, linestyle='--', alpha=0.7)
    ax_pct.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    ax_pct.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Cryptocurrency Trading Volume Analysis (2021-2024)', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig('analysis_results/plots/volume_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_returns_plot(df):
    """Create improved returns comparison plot"""
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Daily Returns Plot
    ax1.plot(df.index, df['btc_price_return'], label='BTC', alpha=0.7)
    ax1.plot(df.index, df['eth_price_return'], label='ETH', alpha=0.7)
    ax1.plot(df.index, df['sol_price_return'], label='SOL', alpha=0.7)
    ax1.set_title('Daily Returns Comparison', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Daily Returns (%)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
    
    # Cumulative Returns Plot
    cum_returns = {
        'BTC': (1 + df['btc_price_return']).cumprod(),
        'ETH': (1 + df['eth_price_return']).cumprod(),
        'SOL': (1 + df['sol_price_return']).cumprod()
    }
    
    for crypto, returns in cum_returns.items():
        ax2.plot(df.index, returns, label=f'{crypto} ({returns.iloc[-1]:.2f}x)', linewidth=2)
    
    ax2.set_title('Cumulative Returns (Multiple of Initial Investment)', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Return (x)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_results/plots/returns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(price_corr, volume_corr, return_corr):
    """Create improved correlation heatmaps"""
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Price correlation heatmap
    sns.heatmap(price_corr, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', ax=axes[0], square=True, vmin=-1, vmax=1)
    axes[0].set_title('Price Correlation', fontsize=12, pad=10)
    
    # Volume correlation heatmap
    sns.heatmap(volume_corr, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', ax=axes[1], square=True, vmin=-1, vmax=1)
    axes[1].set_title('Volume Correlation', fontsize=12, pad=10)
    
    # Returns correlation heatmap
    sns.heatmap(return_corr, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', ax=axes[2], square=True, vmin=-1, vmax=1)
    axes[2].set_title('Returns Correlation', fontsize=12, pad=10)
    
    plt.suptitle('Cryptocurrency Correlations Analysis', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig('analysis_results/plots/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_technical_plots(df):
    """Create improved technical indicator plots"""
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    cryptos = ['btc', 'eth', 'sol']
    
    for crypto in cryptos:
        # Create a figure with subplots for all technical indicators
        fig = plt.figure(figsize=(15, 20))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
        
        # Price and Bollinger Bands Plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df[f'{crypto}_price'], label='Price', color='blue', linewidth=2)
        ax1.plot(df.index, df[f'{crypto}_price_BB_upper'], label='Upper Band', color='red', linestyle='--')
        ax1.plot(df.index, df[f'{crypto}_price_BB_lower'], label='Lower Band', color='green', linestyle='--')
        ax1.fill_between(df.index, df[f'{crypto}_price_BB_upper'], df[f'{crypto}_price_BB_lower'],
                        alpha=0.1, color='gray', label='Bollinger Band Range')
        
        # Add current price and band information
        current_price = df[f'{crypto}_price'].iloc[-1]
        upper_band = df[f'{crypto}_price_BB_upper'].iloc[-1]
        lower_band = df[f'{crypto}_price_BB_lower'].iloc[-1]
        band_width = ((upper_band - lower_band) / current_price) * 100
        
        ax1.annotate(
            f'Current Price: ${current_price:,.0f}\nUpper Band: ${upper_band:,.0f}\nLower Band: ${lower_band:,.0f}\nBand Width: {band_width:.1f}%',
            xy=(df.index[-1], current_price),
            xytext=(10, 10), textcoords='offset points',
            fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        ax1.set_title(f'{crypto.upper()} Price and Bollinger Bands', fontsize=14)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left')
        
        # RSI Plot
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, df[f'{crypto}_price_RSI'], color='blue', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        ax2.fill_between(df.index, 70, df[f'{crypto}_price_RSI'],
                        where=df[f'{crypto}_price_RSI'] >= 70, color='red', alpha=0.3)
        ax2.fill_between(df.index, 30, df[f'{crypto}_price_RSI'],
                        where=df[f'{crypto}_price_RSI'] <= 30, color='green', alpha=0.3)
        
        # Add current RSI value
        current_rsi = df[f'{crypto}_price_RSI'].iloc[-1]
        ax2.annotate(
            f'Current RSI: {current_rsi:.1f}',
            xy=(df.index[-1], current_rsi),
            xytext=(10, 10), textcoords='offset points',
            fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        ax2.set_title(f'{crypto.upper()} Relative Strength Index (RSI)', fontsize=12)
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # MACD Plot
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(df.index, df[f'{crypto}_price_MACD'], label='MACD', linewidth=2)
        ax3.plot(df.index, df[f'{crypto}_price_Signal'], label='Signal', linewidth=2)
        
        # MACD Histogram with color coding
        macd_hist = df[f'{crypto}_price_MACD'] - df[f'{crypto}_price_Signal']
        ax3.bar(df.index, macd_hist, label='MACD Histogram', alpha=0.3,
                color=np.where(macd_hist >= 0, 'g', 'r'))
        
        # Add current MACD values
        current_macd = df[f'{crypto}_price_MACD'].iloc[-1]
        current_signal = df[f'{crypto}_price_Signal'].iloc[-1]
        current_hist = macd_hist.iloc[-1]
        
        ax3.annotate(
            f'MACD: {current_macd:.1f}\nSignal: {current_signal:.1f}\nHistogram: {current_hist:.1f}',
            xy=(df.index[-1], current_macd),
            xytext=(10, 10), textcoords='offset points',
            fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        ax3.set_title(f'{crypto.upper()} MACD', fontsize=12)
        ax3.set_ylabel('MACD', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        # Volume Plot
        ax4 = fig.add_subplot(gs[3])
        volume = df[f'{crypto}_volume']
        ax4.bar(df.index, volume, alpha=0.3, color='blue', label='Volume')
        ax4.set_title(f'{crypto.upper()} Trading Volume', fontsize=12)
        ax4.set_ylabel('Volume', fontsize=10)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        # Add volume statistics
        avg_vol = volume.mean()
        max_vol = volume.max()
        current_vol = volume.iloc[-1]
        
        ax4.annotate(
            f'Current Vol: {current_vol:,.0f}\nAvg Vol: {avg_vol:,.0f}\nMax Vol: {max_vol:,.0f}',
            xy=(df.index[-1], current_vol),
            xytext=(10, 10), textcoords='offset points',
            fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        # Rotate x-axis labels
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{crypto.upper()} Technical Analysis', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(f'analysis_results/plots/{crypto}_technical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_analysis_summary(df, stats, price_corr, volume_corr, return_corr):
    """Save analysis summary to a file"""
    summary = {
        'statistics': stats,
        'correlations': {
            'price': price_corr.to_dict(),
            'volume': volume_corr.to_dict(),
            'returns': return_corr.to_dict()
        },
        'data_info': {
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'total_records': len(df)
        }
    }
    
    with open('analysis_results/data/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

def save_processed_data(df):
    """Save processed data to CSV files"""
    # Save full dataset
    df.to_csv('analysis_results/data/processed_data.csv')
    
    # Save individual cryptocurrency data
    cryptos = ['btc', 'eth', 'sol']
    for crypto in cryptos:
        crypto_cols = [col for col in df.columns if crypto in col]
        df[crypto_cols].to_csv(f'analysis_results/data/{crypto}_processed.csv')

def prepare_data_for_next_steps(df):
    """Prepare data for further analysis"""
    # Calculate additional metrics
    for crypto in ['btc', 'eth', 'sol']:
        # Volatility metrics
        df[f'{crypto}_volatility_30d'] = df[f'{crypto}_price_return'].rolling(window=30).std() * np.sqrt(252)
        df[f'{crypto}_volatility_90d'] = df[f'{crypto}_price_return'].rolling(window=90).std() * np.sqrt(252)
        
        # Price momentum
        df[f'{crypto}_momentum_14d'] = df[f'{crypto}_price'].pct_change(periods=14)
        df[f'{crypto}_momentum_30d'] = df[f'{crypto}_price'].pct_change(periods=30)
        
        # Volume trends
        df[f'{crypto}_volume_ma20'] = df[f'{crypto}_volume'].rolling(window=20).mean()
        df[f'{crypto}_volume_ma50'] = df[f'{crypto}_volume'].rolling(window=50).mean()
        
        # Price ratios
        df[f'{crypto}_price_to_ma50'] = df[f'{crypto}_price'] / df[f'{crypto}_price'].rolling(window=50).mean()
        
    # Save prepared data
    df.to_csv('analysis_results/data/prepared_data.csv')
    
    # Create summary of prepared data
    summary = {
        'metrics_added': [
            'volatility_30d', 'volatility_90d',
            'momentum_14d', 'momentum_30d',
            'volume_ma20', 'volume_ma50',
            'price_to_ma50'
        ],
        'data_shape': df.shape,
        'date_range': {
            'start': df.index.min().strftime('%Y-%m-%d'),
            'end': df.index.max().strftime('%Y-%m-%d')
        }
    }
    
    with open('analysis_results/data/preparation_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return df

def main():
    try:
        # Connect to PostgreSQL
        engine = connect_to_postgres()
        if engine is None:
            return

        # Fetch data
        print("\nFetching data from PostgreSQL...")
        df = fetch_crypto_data(engine)
        if df is None:
            return

        # Calculate basic statistics
        print("\nCalculating basic statistics...")
        df, stats = calculate_basic_stats(df)
        if df is None or stats is None:
            return

        # Calculate correlations
        print("\nCalculating correlations...")
        price_corr, volume_corr, return_corr = calculate_correlations(df)
        if price_corr is None or volume_corr is None or return_corr is None:
            return

        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        df = calculate_technical_indicators(df)
        if df is None:
            return

        # Create visualizations
        print("\nCreating visualizations...")
        create_price_plot(df)
        create_volume_plot(df)
        create_returns_plot(df)
        create_correlation_heatmap(price_corr, volume_corr, return_corr)
        create_technical_plots(df)

        # Save results
        print("\nSaving analysis results...")
        save_analysis_summary(df, stats, price_corr, volume_corr, return_corr)
        save_processed_data(df)

        # Store results in PostgreSQL
        print("\nStoring analysis results in PostgreSQL...")
        store_analysis_results(engine, df, stats, price_corr, volume_corr, return_corr)

        print("\nPreparing data for further analysis...")
        df = prepare_data_for_next_steps(df)

        print("\nAnalysis completed successfully")
        print("\nResults have been saved to:")
        print("- Plots: analysis_results/plots/")
        print("- Data: analysis_results/data/")
        print("- Database: PostgreSQL tables (analysis_results, correlations)")

    except Exception as e:
        print(f"\nError in main function: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {sys.exc_info()}")

if __name__ == "__main__":
    main() 