import streamlit as st
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pyaudio
import wave

# Define the Streamlit app
def app():
    st.title('Infant Cry Classification')
    # Display choice of classifier
    options = ['LSTM', 'Random Forest']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option == 'Random Forest':
        model_path = "myRandomForest.pkl"  # Replace with your model path
    else:
        model_path = "lstm_audio_model.joblib"  # Replace with your model path

    model = joblib.load(model_path)

    def record_audio():
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 5  # Adjust recording duration as needed

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open("recording.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    recorded_audio = st.file_uploader("Upload or record audio", type=["wav", "mp3"])

    if recorded_audio is not None:
        if "wav" in recorded_audio.name:
            with open("recording.wav", "wb") as f:
                f.write(recorded_audio.getbuffer())
        else:
            st.error("Please upload a WAV audio file.")
    elif st.button("Record Audio"):
        record_audio()

    def predict_cry(audio_file):
        try:
            # Preprocess audio (extract MFCC features)
            audio, sr = librosa.load(audio_file)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=1)
            mfcc_scaled = StandardScaler().fit_transform(mfcc.T)
            mfcc_df = pd.DataFrame(mfcc_scaled.T)

            # Make prediction using the loaded model
            prediction = model.predict(mfcc_df)

            # Get the class label
            class_names = model.classes_
            predicted_class = class_names[prediction[0]]

            return predicted_class
        except Exception as e:
            st.error("Error occurred during prediction.")
            return None

    if recorded_audio is not None or st.button("Classify"):
        if recorded_audio is None:
            audio_file = "recording.wav"  # Use recorded audio if available
        else:
            audio_file = "recording.wav"  # Provide the correct path to the audio file

        predicted_cry = predict_cry(audio_file)
        if predicted_cry is not None:
            st.success(f"Predicted cry: {predicted_cry}")

# Run the app
if __name__ == "__main__":
    app()
