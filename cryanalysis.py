# -*- coding: utf-8 -*-
"""CryAnalysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1746vHLVujgyo9qadZL65PrZH1MEDmRjb
"""

import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Define raw audio dictionary
raw_audio = {}

# Loop through directories and label audio files
directories = ['hungry', 'belly_pain', 'burping', 'discomfort', 'tired']
for directory in directories:
    path = '/content/drive/MyDrive/3rd year projects/Research/Thesis/Data and affecting factors/Data Source/donateacry_corpus_cleaned_and_updated_data/' + directory
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            raw_audio[os.path.join(path, filename)] = directory

import wave
import math


def chop_song(filename, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    try:
        handle = wave.open(filename, 'rb')
        frame_rate = handle.getframerate()
        n_frames = handle.getnframes()
        window_size = 2 * frame_rate
        num_secs = int(math.ceil(n_frames/frame_rate))
        last_number_frames = 0
        for i in range(num_secs):
            shortfilename = os.path.basename(filename).split(".")[0]
            snippetfilename = os.path.join(folder, f"{shortfilename}snippet{i+1}.wav")
            snippet = wave.open(snippetfilename ,'wb')
            snippet.setnchannels(2)
            snippet.setsampwidth(handle.getsampwidth())
            snippet.setframerate(frame_rate)
            snippet.setnframes(handle.getnframes())
            snippet.writeframes(handle.readframes(window_size))
            handle.setpos(handle.tell() - 1 * frame_rate)
            if last_number_frames < 1:
                last_number_frames = snippet.getnframes()
            elif snippet.getnframes() != last_number_frames:
                os.rename(snippetfilename, snippetfilename+".bak")
            snippet.close()
        handle.close()
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# # Example usage:
# raw_audio = {'/content/drive/MyDrive/3rd year projects/Research/Thesis/Data and affecting factors/Data Source/donateacry_corpus_cleaned_and_updated_data/hungry/02c3b725-26e4-4a2c-9336-04ddc58836d9-1430726196216-1.7-m-04-hu.wav': 'hungry', '/content/drive/MyDrive/3rd year projects/Research/Thesis/Data and affecting factors/Data Source/donateacry_corpus_cleaned_and_updated_data/belly_pain/643D64AD-B711-469A-AF69-55C0D5D3E30F-1430138495-1.0-m-72-bp.wav': 'belly_pain'}  # Replace with your raw_audio dictionary

for audio_file in raw_audio:
    chop_song(audio_file, raw_audio[audio_file])

# Chop and Transform each track
X = pd.DataFrame(columns=np.arange(45), dtype='float32').astype(np.float32)
for i, audio_file in enumerate(raw_audio.keys()):
    audiofile, sr = librosa.load(audio_file)
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
    x = pd.DataFrame(fingerprint, dtype='float32')
    x[44] = raw_audio[audio_file]
    X.loc[i] = x.loc[0]

  # Handle missing values
X = X.fillna(0)

X.head()

# Split data into train and test sets
y = X[44]
del X[44]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train and evaluate models
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=25, max_features=5)),
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC()),
]

print("Model, Accuracy, Precision, Recall")
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"{model_name}: {accuracy}, {precision}, {recall}")

from sklearn.preprocessing import LabelEncoder

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Prepare data for LSTM
X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define LSTM model
lstm_model = Sequential([
    LSTM(units=128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    Dense(units=64, activation='relu'),
    Dropout(0.2),
    Dense(units=len(np.unique(y_train)), activation='softmax')
])

# Compile LSTM model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train LSTM model
lstm_model.fit(X_train_lstm, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Manually compute predictions
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)

# Calculate accuracy
accuracy_lstm = accuracy_score(y_test_encoded, y_pred_lstm_classes)

"""### Step 2: Making Actual Prediction"""

def pickle_model(model, modelname):
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, str(modelname) + '.pkl'), 'wb') as f:
        return pickle.dump(model, f)

model = RandomForestClassifier()
model.fit(X, y)
pickle_model(model, "myRandomForest")

def getModel(pickle_path):
  with open(pickle_path, 'rb') as f:
        return pickle.load(f)

model = getModel("models/myRandomForest.pkl")

!apt-get install portaudio19-dev
!pip install pyaudio

import wave
import pyaudio

def chop_audio(stream, folder, duration, filename="snippet.wav", format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
  """Records audio from the microphone and saves it to a file in segments.

  Args:
      stream: A pyaudio stream object representing the audio input.
      folder: The directory where the recorded audio segment will be saved.
      duration: The duration (in seconds) for which the audio will be recorded.
      filename (str, optional): The filename of the recorded audio segment. Defaults to "snippet.wav".
      format (int, optional): The sample format of the audio data. Defaults to pyaudio.paInt16 (16-bit signed integer).
      channels (int, optional): The number of audio channels (mono or stereo). Defaults to 1 (mono).
      rate (int, optional): The sampling rate of the audio data in Hz. Defaults to 44100 (CD quality).
      chunk (int, optional): The chunk size for reading audio data from the stream. Defaults to 1024.
  """

  # Create the full filepath
  filepath = os.path.join(folder, filename)

  # Validate filename and folder
  if not os.path.exists(folder):
      os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist

  # Create wave writer object
  with wave.open(filepath, 'wb') as snippet:
      snippet.setnchannels(channels)
      snippet.setsampwidth(pyaudio.get_sample_size(format))
      snippet.setframerate(rate)

      # Slicing Audio stream
      for i in range(0, int(rate / chunk * duration)):
          data = stream.read(CHUNK)
          snippet.writeframes(data)

  # Print confirmation message
  print(f"Audio segment saved to: {filepath}")

# Example usage
p = pyaudio.PyAudio()

# Define format, channels, and rate
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Open audio stream with required arguments
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=1)


chop_audio(stream, "output_folder", 5, filename="my_recording.wav")  # Record for 5 seconds and customize filename

# Close the audio stream
stream.stop_stream()
stream.close()
p.terminate()

predictions = []
for i, filename in enumerate(os.listdir('output_folder/')):
    last_number_frames = -1
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("output_folder/"+filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        prediction = model.predict(fingerprint)
        #print prediction
        predictions.append(prediction[0])

from collections import Counter
data = Counter(predictions)
print(data.most_common() )  # Returns all unique items and their counts
print( data.most_common(1) )