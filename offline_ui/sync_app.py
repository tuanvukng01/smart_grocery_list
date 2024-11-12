from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Connect to MongoDB, add your connection string here
client = MongoClient('')
db = client['grocery_db']  # Database name
grocery_collection = db['grocery_list']

@app.route('/get_list', methods=['GET'])
def get_list():
    # Get all items from MongoDB
    grocery_items = list(grocery_collection.find({}, {'_id': 0}))
    return jsonify(grocery_items)

@app.route('/sync', methods=['POST'])
def sync_list():
    # Sync list from the frontend
    data = request.get_json()
    for item in data.get('grocery_list', []):
        grocery_collection.update_one(
            {'item_name': item['name']},
            {'$set': {'quantity': item['quantity']}},
            upsert=True  # Insert if doesn't exist
        )
    return jsonify({"status": "sync successful"})

@app.route('/add_item', methods=['POST'])
def add_item():
    # Add an item to the grocery list
    data = request.get_json()
    item_name = data.get('name')
    quantity = data.get('quantity')
    grocery_collection.update_one(
        {'item_name': item_name},
        {'$set': {'quantity': quantity}},
        upsert=True
    )
    return jsonify({"status": f"{item_name} added successfully"})

@app.route('/remove_item', methods=['POST'])
def remove_item():
    # Remove an item from the grocery list
    data = request.get_json()
    item_name = data.get('name')
    grocery_collection.delete_one({'item_name': item_name})
    return jsonify({"status": f"{item_name} removed successfully"})

@app.route('/remove_all', methods=['POST'])
def remove_all():
    # Remove all items from the grocery list
    grocery_collection.delete_many({})
    return jsonify({"status": "All items removed successfully"})

@app.route('/get_components', methods=['GET'])
def get_components():
    components = list(db['components'].find({}, {'_id': 0, 'item_name': 1}))
    return jsonify([component['item_name'] for component in components])

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)