import streamlit as st
import pickle
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

# Function to extract features from audio file
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    audio_data, sample_rate = sf.read(file_path)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        features.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).T,axis=0)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T,axis=0)
        features.extend(mel)
    return features

# Load the model and label_encoder
with open('emotion_detection_model.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

# Function to predict emotion of a new audio file
def predict_emotion(audio_file_path):
    # Extract features from the audio file
    features = extract_features(audio_file_path)
    # Convert features to NumPy array
    features = np.array(features)
    # Reshape the features for model input
    features = features.reshape(1, features.shape[0], 1, 1)
    # Make predictions
    predictions = model.predict(features)
    # Decode predictions
    emotion_label = label_encoder.inverse_transform([np.argmax(predictions)])
    return emotion_label[0]

# Streamlit UI
st.title('Emotion Detection from Audio')

# File uploader
audio_file = st.file_uploader("Upload or drag & drop an audio file")
st.warning('Just wait for result. No button's here.\n\n Upload anything other than audio file to see Error.Thanks!')

if audio_file is not None:
    # Predict emotion
    predicted_emotion = predict_emotion(audio_file)
    
    # Display predicted emotion
    if predicted_emotion == 'HAP':
        st.write("Predicted Emotion: Happy")
    elif predicted_emotion == 'ANG':
        st.write("Predicted Emotion: Angry")
    elif predicted_emotion == 'DIS':
        st.write("Predicted Emotion: Disgust")
    elif predicted_emotion == 'NEU':
        st.write("Predicted Emotion: Neutral")
    elif predicted_emotion == 'SAD':
        st.write("Predicted Emotion: Sad")
    elif predicted_emotion == 'FEA':
        st.write("Predicted Emotion: Fear")
