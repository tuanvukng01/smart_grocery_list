# Smart Grocery List Management System

This repository contains the **Smart Grocery List Management System** project, developed as the final project for a Bachelor's degree program. This innovative project utilizes deep learning for image recognition, enabling a Raspberry Pi to identify grocery items and manage a shopping list in real-time.

## Project Overview

The **Smart Grocery List Management System** addresses a common problem: efficiently managing grocery shopping. By combining Raspberry Pi hardware with deep learning and Google Coral USB Accelerator, the system detects and removes items from a digital grocery list as they are placed in the shopping basket, helping users avoid unnecessary purchases and missed items.

The system has two main components:
1. **Edge Device**: A Raspberry Pi 4 with a camera and Google Coral USB Accelerator for image recognition.
2. **Mobile Application**: An app for users to create and sync their grocery lists with the edge device.

## Project Components

1. **Raspberry Pi 4-Based Hardware System**
   - Utilizes a Raspberry Pi 4 with a camera module to capture images.
   - Housed in a custom-designed, 3D-printed enclosure that attaches to a shopping cart.

2. **Google Coral USB Accelerator**
   - Accelerates deep learning inference, enabling the Raspberry Pi to perform image recognition locally.

3. **Deep Learning Model Training**
   - Model is trained in PyTorch to recognize various grocery items.
   - Conversion to TensorFlow Lite allows efficient, on-device processing.
   - A Transfer Learning approach was used with a pre-trained MobileNet model for faster and more accurate training.

4. **Offline Grocery List Management**
   - Recognized items are removed from the grocery list, which is displayed on a touchscreen interface.
   - The system functions offline, relying on an SQLite database to store list data locally.

5. **Mobile Application**
   - Allows users to create and sync grocery lists with the Raspberry Pi system before shopping.

## Deep Learning Process

### Model Training
- **Dataset Collection**: Grocery item images were collected and preprocessed to ensure high-quality, labeled data.
- **Model Selection and Training**: We selected MobileNet, a lightweight convolutional neural network model optimized for edge devices.
- **Transfer Learning**: The model was fine-tuned using PyTorch on our custom dataset to recognize specific grocery items accurately.
- **Conversion to TensorFlow Lite**: After training, the model was converted to TFLite, optimized for real-time inference on resource-constrained devices like the Raspberry Pi.
- **Integration with Google Coral USB Accelerator**: The TFLite model is run on Google Coral to boost recognition speed.

### Inference
- The model performs real-time inference using images captured by the Raspberry Pi camera.
- Each time an item is recognized, it is removed from the list, updating the display in real time.

## Key Features

- **Automatic Item Detection**: Real-time recognition and automatic removal of items from the grocery list.
- **Offline Functionality**: Operates entirely offline, with a local SQLite database and real-time inference without internet connectivity.
- **Mobile Syncing**: Users can manage grocery lists via a mobile app and sync updates with the Raspberry Pi.
- **User-Friendly Interface**: The Raspberry Pi interface provides a clear display of the grocery list and updates it dynamically as items are recognized.

## Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tuanvukng01/smart_grocery_list.git
   cd smart-grocery-list

	2.	Install Dependencies
	•	Make sure the Raspberry Pi has Python, TensorFlow Lite, and PyTorch installed.
	•	Install other required libraries using:

pip install -r requirements.txt


	3.	Set Up Hardware
	•	Attach the camera module and Google Coral USB Accelerator to the Raspberry Pi.
	•	Place the Raspberry Pi in the 3D-printed enclosure (see /hardware_design/ for design files).
	4.	Run the Application
	•	Launch the main application with:

python main.py


	5.	Model Deployment
	•	The pre-trained TFLite model should be placed in the /models/ directory. The app will load the model and start image recognition upon execution.

Usage

	1.	Starting the System: Power on the Raspberry Pi and launch the application.
	2.	Creating a Grocery List: Use the mobile app to create and sync a grocery list with the Raspberry Pi.
	3.	Adding Items: As items are placed in the shopping basket, the camera captures images, and recognized items are removed from the list.
	4.	Real-Time Updates: The grocery list is updated on the touchscreen display in real time.

Documentation

Comprehensive documentation is available within the repository, covering the following:
	•	System Design and Architecture: Detailed design of hardware and software components.
	•	Model Training: Steps taken to train and fine-tune the deep learning model.
	•	User Manual: Guide for setting up, running, and using the system.

Future Enhancements

Potential improvements for this project include:
	•	Extended Item Recognition: Training on a broader dataset for diverse grocery items.
	•	Improved Syncing: Enabling real-time syncing between the mobile app and Raspberry Pi during shopping.
	•	Additional Language Support: Expanding the interface for multi-language compatibility.

Thank you for exploring the Smart Grocery List Management System! This project was a deep dive into deploying AI on edge devices, specifically integrating deep learning to enhance everyday shopping efficiency.
