import streamlit as st
import pickle
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from streamlit_audio_recorder import st_audio_recorder

# Function to extract features from audio file
def extract_features(audio_data, sample_rate, mfcc=True, chroma=True, mel=True):
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

# Function to predict emotion from audio data
def predict_emotion(audio_data, sample_rate):
    # Extract features from the audio data
    features = extract_features(audio_data, sample_rate)
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
st.title('Real-time Emotion Detection from Audio')

# Audio recorder
audio_data, sample_rate = st_audio_recorder(sampling_rate=44100, audio_format="wav")

# Predict emotion when recording is stopped
if st.button("Stop Recording"):
    if audio_data is not None:
        predicted_emotion = predict_emotion(audio_data, sample_rate)
        st.write(f"Predicted Emotion: {predicted_emotion}")
