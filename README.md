Spam Classifier

A Machine Learning-based Spam Classifier that detects whether an email/SMS is spam or ham using TF-IDF vectorization and a Random Forest Classifier.

Features

Clean and preprocess SMS data

Extract text features (TF-IDF) + numeric features (word & character counts)

Train and evaluate Random Forest model

Save and load model with Pickle

Predict new messages in real-time

Dataset

SMS Spam Collection Dataset (spam.csv)

Columns: Category (spam/ham), Message (text content)

Installation
git clone https://github.com/Purushothamcv/SmartSpamDetection.git
cd spam-classifier
pip install -r requirements.txt


Download NLTK data:

import nltk
nltk.download('punkt')

Usage
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Predict
text = "Congratulations! You've won a free ticket!"
X_input = vectorizer.transform([text])
prediction = model.predict(X_input)
print("Prediction:", prediction)

Results

Accuracy: ~96–98%

Easily predicts spam vs ham for new messages

Future Work

Deploy as a web app with Flask/FastAPI

Explore deep learning models for better accuracy

Integrate AI for smarter spam detection
<img width="426" height="421" alt="Screenshot 2025-09-15 022119" src="https://github.com/user-attachments/assets/61cbcf50-0181-44a0-b1d4-55edc6c5519b" />
