# How can I adjust this to use Raspberry pi pyqt5 ?
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QLineEdit, QGridLayout, QMessageBox, QFrame, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
import grocery_list
import grocery_db
import numpy as np
from PIL import Image
import platform
import cv2  # OpenCV for webcam functionality

# Import TensorFlow Lite runtime instead of full TensorFlow
if platform.system() == "Linux":  # Running on a Linux system (e.g., Raspberry Pi)
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow as tflite


class GroceryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.interpreter = self.load_model_with_tpu("uecfood_model.tflite")  # Load model
        self.class_labels = self.load_class_labels("food_list.txt")  # Load class labels
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Grocery List Management')
        self.setGeometry(100, 100, 600, 400)

        # Main layout
        main_layout = QVBoxLayout()

        # Create a frame for the grocery item buttons at the top
        item_frame = QFrame()
        item_frame.setFrameShape(QFrame.StyledPanel)
        item_layout = QGridLayout()
        item_frame.setLayout(item_layout)
        self.grid_layout = item_layout  # Reference for updating items

        # Create a frame for the buttons at the bottom
        button_frame = QFrame()
        button_frame.setFrameShape(QFrame.StyledPanel)
        button_layout = QHBoxLayout()
        button_frame.setLayout(button_layout)

        # Filter search bar to filter dropdown items
        self.filter_entry = QLineEdit()
        self.filter_entry.setPlaceholderText("Search item")
        self.filter_entry.textChanged.connect(self.filter_items)  # Connect the filter

        # Dropdown and quantity entry for adding items
        self.item_dropdown = QComboBox()
        self.item_dropdown.setFixedWidth(180)  # Reduce the width of the dropdown to 60% of its default size
        self.full_component_list = grocery_db.get_components()  # Full list for filtering
        self.item_dropdown.addItems(self.full_component_list)

        self.quantity_entry = QLineEdit()
        self.quantity_entry.setPlaceholderText("Quantity")
        self.quantity_entry.setText("1")

        # Add item button
        add_button = QPushButton("Add Item")
        add_button.clicked.connect(self.add_item)

        # Remove item button
        self.item_entry = QLineEdit()
        self.item_entry.setPlaceholderText("Item name to remove")

        remove_button = QPushButton("Remove Item")
        remove_button.clicked.connect(self.remove_item)

        # Remove all items button
        remove_all_button = QPushButton("Remove All")
        remove_all_button.clicked.connect(self.remove_all_items)

        # Sync button
        sync_button = QPushButton("Sync with MongoDB")
        sync_button.clicked.connect(self.sync_list)

        # Detect and remove button for the AI-based removal
        detect_button = QPushButton("Detect Item and Remove")
        detect_button.clicked.connect(self.detect_and_remove_item)

        # Add buttons to button layout (bottom part)
        button_layout.addWidget(self.filter_entry)  # Add the filter entry at the bottom
        button_layout.addWidget(self.item_dropdown)
        button_layout.addWidget(self.quantity_entry)
        button_layout.addWidget(add_button)
        button_layout.addWidget(self.item_entry)
        button_layout.addWidget(remove_button)
        button_layout.addWidget(remove_all_button)
        button_layout.addWidget(sync_button)
        button_layout.addWidget(detect_button)

        # Add the item frame (top part with items)
        main_layout.addWidget(item_frame)

        # Add spacer to push the buttons to the very bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer)

        # Add the button frame (bottom part with buttons)
        main_layout.addWidget(button_frame)

        # Set the layout for the entire window
        self.setLayout(main_layout)

        # Update the display to show current grocery list items
        self.update_display()

    def update_display(self):
        """Update the grid layout to display grocery items."""
        # Clear the layout
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Fetch the grocery list
        grocery_items = grocery_list.get_list()

        # Add grocery items as buttons in the grid layout
        for idx, (item, quantity) in enumerate(grocery_items):
            button = QPushButton(f"{item}: {quantity}")
            row = idx // 3  # 3 items per row for better alignment
            col = idx % 3
            self.grid_layout.addWidget(button, row, col)

    def add_item(self):
        """Add an item to the grocery list."""
        item_name = self.item_dropdown.currentText()
        quantity = self.quantity_entry.text()

        if item_name and quantity.isdigit():
            grocery_list.add_item(item_name, int(quantity))
            self.update_display()
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid quantity.")

    def remove_item(self):
        """Remove an item from the grocery list."""
        item_name = self.item_entry.text()

        if item_name:
            grocery_list.remove_item(item_name)
            self.update_display()
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid item name to remove.")

    def remove_all_items(self):
        """Remove all items from the grocery list."""
        grocery_list.remove_all_items()
        self.update_display()

    def sync_list(self):
        """Sync the grocery list with MongoDB."""
        sync_status = grocery_list.sync_with_mongo()
        QMessageBox.information(self, "Sync Status", sync_status)
        self.update_display()

    def filter_items(self):
        """Filter the items in the dropdown based on search input."""
        filter_text = self.filter_entry.text().lower()
        self.item_dropdown.clear()

        # Filter and display matching items
        filtered_items = [item for item in self.full_component_list if filter_text in item.lower()]
        self.item_dropdown.addItems(filtered_items)

    def detect_and_remove_item(self):
        """Capture an image, classify it, and remove the detected item."""
        image = self.capture_image_from_webcam()
        if image is None:
            QMessageBox.warning(self, "Capture Failed", "Image capture failed.")
            return

        result = self.classify_image(image)
        predicted_class_idx = np.argmax(result)
        predicted_class_label = self.class_labels.get(predicted_class_idx, "Unknown")

        if predicted_class_label != "Unknown":
            grocery_items = grocery_list.get_list()
            for item, quantity in grocery_items:
                if item == predicted_class_label:
                    if quantity > 1:
                        grocery_list.add_item(predicted_class_label, -1)
                    else:
                        grocery_list.remove_item(predicted_class_label)
                    self.update_display()
                    return
            QMessageBox.information(self, "Item Not Found", f"Detected item {predicted_class_label} not found.")
        else:
            QMessageBox.warning(self, "Unknown Item", "Detected item is unknown.")

    def load_model_with_tpu(self, model_path):
        """Load the TFLite model."""
        if platform.system() == "Linux":
            # Use the Edge TPU delegate for Raspberry Pi with Coral
            delegate = [tflite.lite.load_delegate('libedgetpu.so.1')]
            interpreter = tflite.lite.Interpreter(model_path=model_path, experimental_delegates=delegate)
        else:
            # Use the standard TFLite interpreter on other platforms
            interpreter = tflite.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def load_class_labels(self, file_path):
        """Load class labels from a file."""
        class_labels = {}
        with open(file_path, 'r') as file:
            for line in file.readlines()[1:]:
                class_id, class_name = line.strip().split('\t')
                class_labels[int(class_id) - 1] = class_name
        return class_labels

    def capture_image_from_webcam(self):
        """Capture an image from the webcam."""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        cap.release()
        return image

    def classify_image(self, image):
        """Classify the captured image."""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_data = self.preprocess_image(image)
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def preprocess_image(self, image):
        """Preprocess the image for model inference."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.expand_dims(image, axis=0)
        return image


def main():
    app = QApplication(sys.argv)
    window = GroceryApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    grocery_list.create_list()
    main()