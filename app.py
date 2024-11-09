import copy
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load pre-trained model from TensorFlow Hub
MODEL_DIR = "models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model"
detector = tf.saved_model.load(MODEL_DIR)

# Load labels for COCO dataset
LABELS_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt"
labels_path = "coco_labels.txt"
if not os.path.exists(labels_path):
    import requests
    r = requests.get(LABELS_URL)
    with open(labels_path, 'w') as f:
        f.write(r.text)

with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to run object detection on an image
def delete_previous_image(upload_folder):
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete the file
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# Object detection and cut-out function
def detect_objects_and_cutout(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
    img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension

    # Perform detection
    detections = detector(img_tensor)

    # Extract detection results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    confidence_threshold = 0.5
    detected_objects = []

    cutout_paths = []  # To store paths of individual object cutouts
    processed_img = copy.deepcopy(img)

    for i in range(num_detections):
        score = detections['detection_scores'][i]
        if score < confidence_threshold:
            continue

        box = detections['detection_boxes'][i]
        class_id = detections['detection_classes'][i]
        class_name = labels[class_id - 1] if class_id <= len(labels) else 'N/A'

        # Get the bounding box coordinates
        height, width, _ = img.shape
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))

        # Cut out the object from the image
        cutout = img[top:bottom, left:right]
        cutout_filename = f"cutout_{class_name}_{i}.jpg"
        cutout_path = os.path.join(app.config['UPLOAD_FOLDER'], cutout_filename)
        cv2.imwrite(cutout_path, cutout)
        cutout_paths.append((cutout_filename, class_name))
        detected_objects.append(f"{class_name} ({int(score * 100)}%)")

         
        # Draw bounding box on the original image
        cv2.rectangle(processed_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(processed_img, f"{int(score * 100)}%", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Save the image with bounding boxes
    processed_image_file_name = "processed_" + os.path.basename(image_path)
    processed_image_path = os.path.join(UPLOAD_FOLDER, processed_image_file_name)
    cv2.imwrite(processed_image_path, processed_img)

    return detected_objects, cutout_paths, processed_image_file_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Delete the previous uploaded image before saving the new one
            delete_previous_image(app.config['UPLOAD_FOLDER'])

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Perform object detection and cut out objects
            detected_objects, cutout_paths, processed_image_path = detect_objects_and_cutout(file_path)

            # Redirect to display the detected objects and cutouts
            return render_template('index.html',filename=filename, processed_image_path=processed_image_path, detected_objects=detected_objects, cutout_paths=cutout_paths)
    return render_template('index.html', filename=None, processed_image_path=None, detected_objects=[], cutout_paths=[], )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
