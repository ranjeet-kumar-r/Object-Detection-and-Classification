# Object Detection System

## Overview
This project features an object detection system powered by the Faster R-CNN ResNet50 model. It detects and classifies objects in images, draws bounding boxes, and displays confidence scores for each detected object. Built with Flask as the backend, the application utilizes Python libraries such as TensorFlow and OpenCV for efficient image processing and object detection.


## Features
- Upload images for object detection and classification.
- Detect and classify multiple objects within the uploaded image.
- Display bounding boxes around detected objects.
- Show confidence scores for each detected object.
- Preview original and processed images side-by-side.
- List and display cut-out images of detected objects.
- Support for common image file formats (PNG, JPG, JPEG, GIF).
- Responsive web interface built with Bootstrap.
- Automatic label download and integration with COCO dataset.
- Secure and validated image upload handling.

## Requirements
To run this project, you need to have the following installed:
- **Python 3.x**
- **pip** (Python package installer)

### Optional Tools:
- **Virtual environment** (recommended but not mandatory)

## Setup Instructions

Clone the repository from GitHub to your local machine:

```bash
git clone https://github.com/yourusername/object-detection-faster-rcnn.git
cd object-detection-faster-rcnn
```

## Model Download
This project uses the pre-trained Faster R-CNN ResNet50 model trained on the COCO dataset. You can download the model from the TensorFlow Model Zoo using the following link:

- [Download Faster R-CNN ResNet50 (COCO)](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)

After downloading the model, extract the files and place the model in the `models/` directory of the project.

### Command to download the model:
```
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
mkdir models
tar -xvzf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz -C models/
```


### Model Directory Structure:
```bash
models/
├── faster_rcnn_resnet50
```

## Environment Setup (and Why It's Important)

### What is a Virtual Environment?
A virtual environment is a self-contained directory that contains a Python installation for a specific version of Python, plus a number of additional packages. Virtual environments allow you to manage dependencies for projects more easily, ensuring that each project has its own set of installed libraries, independent of others.

### Why Use a Virtual Environment?
- **Avoid Dependency Conflicts**: Different projects may require different versions of the same package, and managing these versions globally on your machine can lead to conflicts.
- **Cleaner Project**: Each project will have its own environment, reducing the risk of affecting other Python projects on your system.
- **Easy to Manage**: It's easier to manage, install, and upgrade packages specific to your project.

While not mandatory, using a virtual environment is highly recommended to keep your development environment isolated.

## Set Up a Virtual Environment (Optional but Recommended)

Setting up a virtual environment is recommended to avoid conflicts with other Python packages on your system. Here’s how to set it up:

### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### For Windows:
```bash
 python -m venv venv
 venv\Scripts\activate
```

After activating the virtual environment, your terminal should show `(venv)` before the command prompt, indicating the virtual environment is active.

## Install Dependencies
After cloning the repository (and activating the virtual environment if you chose to set it up), install the required Python dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
This will install all necessary libraries like Flask, TensorFlow, OpenCV, NumPy, etc.

## Run the Application
To start the Flask application, you can use one of the following commands:

### Using flask run:
```
flask run
```
### Using python app.py:
```
python app.py 
```
Once the server is running, open your web browser and navigate to http://127.0.0.1:5000/. Here, you can upload images, and the object detection results will be displayed.