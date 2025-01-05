# Cryptocurrency Analysis Framework

## Overview
This project is a comprehensive cryptocurrency analysis framework that processes and analyzes historical data for Bitcoin (BTC), Ethereum (ETH), and Solana (SOL). It implements various technical analysis methods and machine learning models for cryptocurrency market analysis.

### Key Features
- Historical data collection from multiple sources (CoinAPI, Yahoo Finance)
- Dual database system (MongoDB for raw data, PostgreSQL for analysis)
- Technical analysis indicators calculation
- Machine learning predictions using LSTM and Prophet models
- Data verification and integrity checks

## Project Structure
```
├── Data Collection Scripts
│   ├── get_historical_data.py    # Fetches BTC/ETH data from CoinAPI
│   ├── get_solana_data.py        # Fetches SOL data from Yahoo Finance
│   └── get_coinapi_symbols.py    # Gets available trading pairs info
│
├── Database Scripts
│   ├── store_crypto_data.py      # Stores raw data in MongoDB
│   ├── store_to_postgres.py      # Transfers data to PostgreSQL
│   ├── verify_mongo_data.py      # Verifies MongoDB data integrity
│   └── verify_postgres_data.py   # Verifies PostgreSQL data integrity
│
├── Analysis Scripts
│   ├── crypto_analysis.py        # Main technical analysis
│   ├── crypto_dec_2024.py        # ML model implementation
│   ├── export_for_ml.py          # Prepares data for ML
│   └── verify_analysis.py        # Validates analysis results
│
├── Data Files
│   ├── BITSTAMP_SPOT_BTC_USD_*.json
│   ├── BITSTAMP_SPOT_ETH_USD_*.json
│   └── SOLANA_SOL_USD_*.csv
│
└── Output Directories
    ├── analysis_results/         # Analysis output files
    └── ml_data/                  # Machine learning datasets
```

## Prerequisites
- Python 3.8 or higher
- MongoDB 6.0 or higher
- PostgreSQL 14.0 or higher
- CoinAPI API key (for data collection)

## Installation

1. **Clone the Repository**
```bash
git clone [repository-url]
cd [repository-name]
```

2. **Set Up Virtual Environment**
```bash
python -m venv .venv
# For Windows
.venv\Scripts\activate
# For Unix/MacOS
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Database Setup**

MongoDB Setup:
```bash
1. Install MongoDB Compass
2. Create database: crypto_analysis
3. Create collections:
   - btc_data
   - eth_data
   - sol_data
   - technical_indicators
```

PostgreSQL Setup:
```bash
1. Install PostgreSQL
2. Create database: crypto_analysis
3. Configure connection in scripts:
   - Default credentials: postgres:11225
   - Default port: 5432
```

## Usage

1. **Data Collection**
```bash
# Collect BTC and ETH data
python get_historical_data.py

# Collect Solana data
python get_solana_data.py

# Get trading pairs information
python get_coinapi_symbols.py
```

2. **Data Storage**
```bash
# Store in MongoDB
python store_crypto_data.py

# Transfer to PostgreSQL
python store_to_postgres.py
```

3. **Run Analysis**
```bash
# Perform technical analysis
python crypto_analysis.py

# Run ML predictions
python crypto_dec_2024.py

# Export data for ML
python export_for_ml.py
```

4. **Verify Results**
```bash
# Verify database integrity
python verify_mongo_data.py
python verify_postgres_data.py

# Verify analysis results
python verify_analysis.py
```

## Output
- Analysis results are stored in the `analysis_results/` directory
- ML datasets are stored in the `ml_data/` directory
- Logs and verification reports are generated for each process

## Error Handling
- All scripts include comprehensive error handling
- Check log files for detailed error messages
- Database verification scripts ensure data integrity

## License
MIT License

## Contributing
Contributions are welcome! Please read the contributing guidelines before making any changes.

## Support
For support or questions, please open an issue in the repository.
