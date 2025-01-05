import requests
import json
from datetime import datetime

def get_all_symbols():
    # API endpoint
    url = "https://rest.coinapi.io/v1/symbols"
    
    # Headers with API key
    headers = {
        'Accept': 'application/json',
        'X-CoinAPI-Key': '3683B432-DC78-4C48-8B12-A8BD5A8FC2D4'
    }
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the JSON response
            symbols_data = response.json()
            
            # Save the data to a JSON file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'symbols_data_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(symbols_data, f, indent=4)
            
            print(f"Successfully retrieved {len(symbols_data)} symbols")
            print(f"Data saved to {filename}")
            return symbols_data
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    print("Fetching symbols data from CoinAPI...")
    symbols = get_all_symbols() 