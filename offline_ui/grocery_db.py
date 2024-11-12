import sqlite3
import os

DB_PATH = 'grocery_list.db'
COMPONENTS_FILE = 'food_list.txt'

def init_db():
    # Initialize the database and create tables if they don't exist
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create grocery list table (active list)
    c.execute('''CREATE TABLE IF NOT EXISTS grocery_list
                 (id INTEGER PRIMARY KEY, item_name TEXT, quantity INTEGER)''')

    # Create components table (256 predefined items)
    c.execute('''CREATE TABLE IF NOT EXISTS components
                 (id INTEGER PRIMARY KEY, item_name TEXT)''')

    # Check if the components table is empty, and if so, populate it
    c.execute("SELECT COUNT(*) FROM components")
    if c.fetchone()[0] == 0:
        components = parse_components_file(COMPONENTS_FILE)
        c.executemany("INSERT INTO components (id, item_name) VALUES (?, ?)", components)

    conn.commit()
    conn.close()

def parse_components_file(file_path):
    """
    Parse the components.txt file to extract (id, name) pairs.
    :param file_path: Path to the components file.
    :return: A list of tuples containing (id, name) for each component.
    """
    components = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                # Split each line by whitespace, assuming the format: id <tab> name
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    component_id = int(parts[0].strip())  # Convert id to integer
                    component_name = parts[1].strip()  # Component name as string
                    components.append((component_id, component_name))
    return components


def add_item_to_db(item_name, quantity):
    # Add an item to the active grocery list or update its quantity if it already exists
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if item already exists
    c.execute("SELECT quantity FROM grocery_list WHERE item_name = ?", (item_name,))
    result = c.fetchone()

    if result:
        # Item already exists, update quantity
        current_quantity = result[0]
        new_quantity = current_quantity + quantity
        print(f"Updating quantity of {item_name} from {current_quantity} to {new_quantity}")  # Debug statement
        c.execute("UPDATE grocery_list SET quantity = ? WHERE item_name = ?", (new_quantity, item_name))
    else:
        # Item doesn't exist, insert new entry
        print(f"Inserting item into DB: {item_name}, Quantity: {quantity}")  # Debug statement
        c.execute("INSERT INTO grocery_list (item_name, quantity) VALUES (?, ?)", (item_name, quantity))

    conn.commit()
    conn.close()

def remove_item_from_db(item_name):
    # Remove an item from the active grocery list
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM grocery_list WHERE item_name = ?", (item_name,))
    conn.commit()
    conn.close()

def get_list_from_db():
    # Get the current active grocery list
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT item_name, quantity FROM grocery_list")
    items = c.fetchall()
    conn.close()
    return items

def get_components():
    # Get the list of 256 predefined components
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT item_name FROM components")
    components = [row[0] for row in c.fetchall()]
    conn.close()
    return components

def remove_all_items_from_db():
    # Remove all items from the active grocery list
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    print("Removing all items from DB...")  # Debug statement
    c.execute("DELETE FROM grocery_list")  # Delete all rows
    conn.commit()
    conn.close()