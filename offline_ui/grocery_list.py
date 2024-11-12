import requests
import grocery_db

def create_list():
    # Initialize the list (if needed)
    grocery_db.init_db()

def add_item(item_name, quantity):
    # Add an item to the grocery list
    grocery_db.add_item_to_db(item_name, quantity)

def remove_item(item_name):
    # Remove an item from the grocery list
    grocery_db.remove_item_from_db(item_name)

def get_list():
    # Retrieve the current grocery list
    grocery_list = grocery_db.get_list_from_db()
    return grocery_list

def remove_all_items():
    # Remove all items from the grocery list
    grocery_db.remove_all_items_from_db()

def sync_with_mongo():
    # Sync the local list with the MongoDB list (fetched via Flask API)
    try:
        response = requests.get('http://localhost:5001/get_list')  # Ensure Flask backend is running on port 5000
        if response.status_code == 200:
            # Remove all local items
            remove_all_items()

            # Add the fetched items to the local list
            remote_items = response.json()  # Assuming Flask returns a list of item_name and quantity pairs
            for item in remote_items:
                add_item(item['item_name'], item['quantity'])
            return "Sync successful"
        else:
            return f"Failed to sync. Status code: {response.status_code}"
    except Exception as e:
        return f"Error during sync: {str(e)}"