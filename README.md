Smart Waste AI is an AI-powered real-time waste detection and classification system designed to promote efficient waste management and environmental sustainability. Utilizing computer vision and deep learning, the system captures input from a webcam and classifies waste into biodegradable and non-biodegradable categories.

Smart Waste AI
Smart Waste AI is an intelligent waste detection and classification system that uses real-time computer vision to sort waste into biodegradable and non-biodegradable categories. This project leverages AI to promote efficient waste management and environmental sustainability.

🔍 Features
Real-time waste detection using webcam input
Deep learning-based object detection (YOLOv3 / MobileNet SSD)
Classification of waste into biodegradable and non-biodegradable
Live visualization with bounding boxes and labels
Easily extendable for more waste categories
🧠 Technologies Used
Python
OpenCV
TensorFlow / Keras
YOLOv3 / MobileNet SSD (pre-trained models)
NumPy
📂 Project Structure
waste_classification_project/ │ ├── data/ # Dataset storage │ ├── biodegradable/ # Images of biodegradable waste │ └── non_biodegradable/ # Images of non-biodegradable waste │ ├── model/ # Saved models │ └── waste_classifier.h5 # Trained Keras classification model │ ├── source/ # Source environment (optional) │ ├── Include/ │ ├── Lib/ │ ├── Scripts/ │ └── pyvenv.cfg │ ├── venv/ # Virtual environment │ ├── Include/ │ ├── Lib/ │ ├── Scripts/ │ └── pyvenv.cfg │ ├── yolov3/ # YOLOv3 configuration and weights │ ├── coco.names # Class names │ ├── yolov3.cfg # YOLOv3 config │ └── yolov3.weights # YOLOv3 pretrained weights │ ├── classify.py # Classifies cropped objects from YOLO ├── predict.py # End-to-end prediction using YOLO + classifier ├── train_model.py # Trains the Keras classification model │ ├── requirements.txt # Python dependencies └── README.md # Project overview & setup instructions

The Final OUTPUT is Screenshot 2025-05-05 082722 Screenshot 2025-05-05 082145
