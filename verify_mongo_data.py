from pymongo import MongoClient

def check_mongodb_data():
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['CrytoAnalysis']
        
        # Check all collections
        print("\nAvailable collections:", db.list_collection_names())
        
        # Check data in each collection
        collections = ['symbols', 'BTC', 'ETH', 'SOL']
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"\n{collection_name} Collection:")
            print(f"Number of documents: {count}")
            
            if count > 0:
                # Show sample document
                sample = collection.find_one()
                print("\nSample document:")
                if collection_name == 'symbols':
                    # For symbols, show specific fields of interest
                    print({
                        'symbol_id': sample.get('symbol_id'),
                        'exchange_id': sample.get('exchange_id'),
                        'asset_id_base': sample.get('asset_id_base'),
                        'asset_id_quote': sample.get('asset_id_quote')
                    })
                else:
                    print(sample)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Checking MongoDB data...")
    check_mongodb_data() 