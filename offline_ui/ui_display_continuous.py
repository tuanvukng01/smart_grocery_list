import time
import tkinter as tk
import grocery_list
import grocery_db
import numpy as np
import tensorflow as tf  # Use TensorFlow for model inference
from PIL import Image
import platform
import cv2  # Import OpenCV for webcam functionality


def recreate_ui(root):
    """
    Destroy the current UI and re-render it from scratch.
    """
    root.destroy()  # Destroy the current window
    create_ui()  # Recreate the entire UI


def update_display_buttons(item_frame):
    """
    Dynamically create buttons to display grocery items and arrange them in two columns.
    """
    # Clear all existing widgets in the frame
    for widget in item_frame.winfo_children():
        widget.destroy()

    # Fetch grocery list from database
    grocery_items = grocery_list.get_list()
    print(f"Updating buttons for items: {grocery_items}")  # Debug statement

    # Display items in two columns
    for idx, (item, quantity) in enumerate(grocery_items):
        display_string = f"{item}: {quantity}"  # Format the string

        # Create button for the current item
        item_button = tk.Button(item_frame, text=display_string, relief="raised")

        # Arrange buttons in two columns
        row = idx // 2  # Determine the row
        col = idx % 2  # Determine the column (0 or 1)
        item_button.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")  # Expand to fill grid cell

    # Configure the rows and columns of the grid to ensure proper resizing
    for i in range((len(grocery_items) + 1) // 2):  # Configure rows for all buttons
        item_frame.grid_rowconfigure(i, weight=1)
    for j in range(2):  # There are always 2 columns
        item_frame.grid_columnconfigure(j, weight=1)


# Load the TFLite model (without TPU delegate for testing on Mac)
def load_model_with_tpu(model_path):
    if platform.system() == "Linux":  # Check if it's running on Raspberry Pi
        delegate = [tf.lite.experimental.load_delegate('libedgetpu.so.1')]  # TPU delegate for Edge TPU on Pi
        interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegate)
    else:  # Mac or other systems
        interpreter = tf.lite.Interpreter(model_path=model_path)  # Regular TFLite interpreter
    interpreter.allocate_tensors()
    return interpreter


# Function to preprocess the input image
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image


# Function to perform classification on the captured image
def classify_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


# Load the class labels
def load_class_labels(file_path):
    class_labels = {}
    with open(file_path, 'r') as file:
        for line in file.readlines()[1:]:
            class_id, class_name = line.strip().split('\t')
            class_labels[int(class_id) - 1] = class_name
    return class_labels


def process_video_stream(interpreter, class_labels, root):
    """
    Process video stream continuously and detect items.
    When an item is detected, it updates the grocery list and waits for 2 seconds before resuming.
    """
    cap = cv2.VideoCapture(0)  # Initialize webcam
    last_detection_time = time.time()  # Track time for the delay after detection

    while True:
        ret, frame = cap.read()  # Read from the webcam
        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Convert the captured frame to RGB for PIL processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        result = classify_image(interpreter, image)
        predicted_class_idx = np.argmax(result)
        predicted_class_label = class_labels.get(predicted_class_idx, "Unknown")

        if predicted_class_label != "Unknown":
            print(f"Detected item: {predicted_class_label}. Adjusting quantity in grocery list.")

            grocery_items = grocery_list.get_list()

            # Find the detected item in the grocery list
            for item, quantity in grocery_items:
                if item == predicted_class_label:
                    if quantity > 1:
                        # Reduce quantity by 1
                        grocery_list.add_item(predicted_class_label, -1)
                        print(f"Decreased quantity of {predicted_class_label} to {quantity - 1}")
                    else:
                        # Remove item if quantity is 1
                        grocery_list.remove_item(predicted_class_label)
                        print(f"Removed {predicted_class_label} from grocery list.")
                    break

            # Recreate the UI after modification
            recreate_ui(root)

            # Wait for 2 seconds before continuing detection
            time.sleep(2)

        # To prevent overloading the CPU, we can add a small delay between frames
        cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()


def add_item_ui(item_dropdown, quantity_entry, root):
    """
    Add the selected item from the dropdown to the grocery list.
    """
    item_name = item_dropdown.get()  # Get the selected item from the dropdown
    quantity = quantity_entry.get()  # Get the quantity from the input

    print(f"Adding item: {item_name} with quantity: {quantity}")  # Debug statement

    if item_name and quantity.isdigit():  # Ensure valid item and quantity
        print(f"Valid input. Adding {item_name} to the list.")  # Debug statement for valid input
        grocery_list.add_item(item_name, int(quantity))  # Add item to the grocery list
        recreate_ui(root)  # Recreate the entire UI after adding an item
    else:
        print("Invalid quantity. Please enter a valid number.")  # Debug statement for invalid input


def remove_item_ui(item_entry, root):
    """
    Remove the entered item from the grocery list.
    """
    item_name = item_entry.get()  # Get the item name from the entry box

    print(f"Removing item: {item_name}")  # Debug statement

    grocery_list.remove_item(item_name)  # Remove the item
    recreate_ui(root)  # Recreate the entire UI after removing an item


def remove_all_ui(root):
    """
    Remove all items from the grocery list.
    """
    print("Removing all items from the list...")  # Debug statement
    grocery_list.remove_all_items()  # Call function to remove all items
    recreate_ui(root)  # Recreate the entire UI after removing all items


def sync_ui(root):
    """
    Sync local grocery list with the MongoDB list via Flask API.
    """
    print("Syncing with MongoDB...")  # Debug statement
    sync_status = grocery_list.sync_with_mongo()  # Sync with MongoDB
    print(sync_status)  # Output the result of the sync
    recreate_ui(root)  # Recreate the UI to reflect the updated list


def create_ui():
    """
    Initialize and run the UI.
    """
    print("Creating UI...")  # Debug statement
    root = tk.Tk()
    root.title("Grocery List Management")

    # Set window size and position (e.g., 400x300 at position (100, 100))
    root.geometry("600x400+100+100")  # Set the window size and position

    # Create a frame to hold the dynamic buttons for the grocery items (top section)
    item_frame = tk.Frame(root)
    item_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # Dropdown menu for selecting items (load components from the database)
    components = grocery_db.get_components()  # Fetch the list of components
    print(f"Loaded components: {components}")  # Debug statement

    # Control frame for inputs and buttons (bottom section)
    control_frame = tk.Frame(root)
    control_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    # Configure columns for uniform size
    control_frame.grid_columnconfigure(0, weight=1)
    control_frame.grid_columnconfigure(1, weight=1)
    control_frame.grid_columnconfigure(2, weight=1)
    control_frame.grid_columnconfigure(3, weight=1)
    control_frame.grid_columnconfigure(4, weight=1)

    # Input section for adding items
    item_dropdown = tk.StringVar(root)
    item_dropdown.set(components[0])  # Set the default value
    dropdown_menu = tk.OptionMenu(control_frame, item_dropdown, *components)  # Create dropdown
    dropdown_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    # Entry for quantity
    quantity_entry = tk.Entry(control_frame, width=10)
    quantity_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    quantity_entry.insert(0, "1")  # Default quantity

    # Sync button
    sync_button = tk.Button(control_frame, text="Sync", command=lambda: sync_ui(root))
    sync_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

    # Add item button
    add_button = tk.Button(control_frame, text="Add Item",
                           command=lambda: add_item_ui(item_dropdown, quantity_entry, root))
    add_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

    # Entry for removing items (manual input)
    item_entry = tk.Entry(control_frame, width=20)
    item_entry.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    item_entry.insert(0, "Item name")  # Default text for removing an item

    # Remove item button
    remove_button = tk.Button(control_frame, text="Remove Item", command=lambda: remove_item_ui(item_entry, root))
    remove_button.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

    # Remove all items button
    remove_all_button = tk.Button(control_frame, text="Remove All", command=lambda: remove_all_ui(root))
    remove_all_button.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

    # Add "Start Detection" button to start the video stream
    model_path = "uecfood_model_edgetpu.tflite"  # Model path
    class_label_file = "food_list.txt"  # Class labels path
    interpreter = load_model_with_tpu(model_path)  # Load the model
    class_labels = load_class_labels(class_label_file)  # Load class labels
    detect_button = tk.Button(control_frame, text="Start Detection",
                              command=lambda: process_video_stream(interpreter, class_labels, root))
    detect_button.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

    # Initial display update (display current items as buttons)
    update_display_buttons(item_frame)

    # Configure row/column weights to make item_frame expand with window resize
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    item_frame.grid_rowconfigure(0, weight=1)
    item_frame.grid_columnconfigure(0, weight=1)

    root.mainloop()


if __name__ == "__main__":
    print("Initializing grocery list...")  # Debug statement
    grocery_list.create_list()  # Initialize the grocery list
    create_ui()  # Run the UI