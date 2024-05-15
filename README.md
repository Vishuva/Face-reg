import numpy as np
import cv2
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# Load FaceNet model
facenet_model = load_model('facenet_keras.h5')

# Load MTCNN detector
detector = MTCNN()

# Load SVM classifier (or any other classifier)
classifier = SVC(kernel='linear', probability=True)
classifier.load_model('svm_classifier.pkl')

# Load label encoder (if needed)
# label_encoder = LabelEncoder()
# label_encoder.load_model('label_encoder.pkl')

# Load normalizer for face embeddings
normalizer = Normalizer()
normalizer.load_model('normalizer.pkl')

# Function to preprocess face image
def preprocess_face(image, required_size=(160, 160)):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to required size
    image = cv2.resize(image, required_size)
    
    # Convert pixel values to float
    image = image.astype('float32')
    
    # Standardize pixel values
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    
    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)
    
    return image

# Function to extract face embeddings
def get_face_embeddings(image):
    # Detect faces in the image
    faces = detector.detect_faces(image)
    
    # Extract embeddings for each face
    embeddings = []
    for face in faces:
        # Get coordinates of the bounding box
        x, y, w, h = face['box']
        
        # Extract face ROI (Region of Interest)
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess face image
        face_roi = preprocess_face(face_roi)
        
        # Generate face embeddings using FaceNet model
        embedding = facenet_model.predict(face_roi)
        
        # Normalize embeddings
        embedding = normalizer.transform(embedding)
        
        embeddings.append(embedding)
    
    return embeddings

# Function to recognize faces
def recognize_faces(image):
    # Extract face embeddings
    embeddings = get_face_embeddings(image)
    
    # Classify faces using SVM classifier
    predictions = classifier.predict(embeddings)
    
    # Get class probabilities
    probabilities = classifier.predict_proba(embeddings)
    
    # Get class labels (if needed)
    # class_labels = label_encoder.inverse_transform(predictions)
    
    return predictions, probabilities

# Load test image
test_image = cv2.imread('test_image.jpg')

# Perform facial recognition
predictions, probabilities = recognize_faces(test_image)

# Display results
for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
    # Get predicted class label (if needed)
    # label = class_labels[i]
    
    # Get class probability
    prob = probability[prediction]
    
    # Display prediction and probability
    print(f'Face {i+1}: Prediction - {prediction}, Probability - {prob:.2f}')
