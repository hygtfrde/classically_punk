import os
import datetime
import ast
import json
import importlib.util

import librosa
from scipy.signal import spectrogram
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.callbacks import TensorBoard

from audio_vizualizer import AudioDataVisualizer
from music_processor import MusicDataProcessor
from genre_classifier_model import MusicGenreClassifier

"""
    DATA COLUMNS SUMMARY
    - filename: The name of the audio file.
    - genre: The genre of the audio file.
    - mfcc: Mel-Frequency Cepstral Coefficients, which represent the short-term power spectrum of the audio file.
    - chroma: Chroma features, which relate to the twelve different pitch classes.
    - mel: Mel spectrogram, which represents the power of a signal in the mel scale frequencies.
    - contrast: Spectral contrast, which measures the difference in amplitude between peaks and valleys in a sound spectrum.
    - tonnetz: Tonnetz features, which capture harmonic and tonal properties.
    - harmony: Harmonic features of the audio.
    - perceptr: Perceptual features.
    - tempo: The tempo of the audio file.
"""

# ------------------------------- GLOBAL VARS
test_audio_file_1 = 'genres/blues/blues.00000.wav'
BLUE = '\033[34m'
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'



# ------------------------------- HELPERS
def get_user_input(prompt, default_value=True):
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'n']:
            return response == 'y'
        elif response == '':
            return default_value
        else:
            print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")


def prompt_for_gpu():
    response = input("Do you want to use GPU for training if available? (Y/N): ").strip().lower()
    if response == 'y':
        script_path = os.path.join('scripts', 'config_tf.py')
        if os.path.exists(script_path):
            spec = importlib.util.spec_from_file_location("config_tf", script_path)
            config_tf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_tf)
            print("GPU setup completed successfully.")
            print('To manage GPU usage run: \n tensorboard --logdir=logs/fit')
        else:
            print("GPU configuration script not found.")
    else:
        print("Using CPU defaults for training.")
        tf.config.set_visible_devices([], 'GPU')




# ------------------------------- MAIN -------------------------------
def main():
    print("Configure the following options:")
    process_data = get_user_input("Do you want to process the data? (Y/N): ", default_value=True)
    visualize_data = get_user_input("Do you want to visualize data? (Y/N): ", default_value=True)
    train_model = get_user_input("Do you want to train the model? (Y/N): ", default_value=True)
    predict_genre = get_user_input("Do you want to predict genre? (Y/N): ", default_value=True)
    
    
    # ------------------------------- MusicDataProcessor
    music_data = None
    default_csv_file_path = 'df_output/test.csv'
    if process_data:
        print(f"{BLUE}Begin Data Processing{RESET}")
        dataset_path = 'genres'
        genre_classifier = MusicDataProcessor(dataset_path, 1, 'just_1_file')

        print("Loading data...")
        genre_classifier.load_data()
        print("Data loaded successfully and validated.")

        print("Getting data...")
        music_data = genre_classifier.get_data()
        print('Music Data for Processor: \n', music_data)
        
    else: print('Skipping Data Processing')
    
    
    # ------------------------------- AudioDataVisualizer
    if visualize_data:
        if music_data is None:
            print('No currently processed data. Using Default DF for music data')
            if not os.path.exists(default_csv_file_path):
                print(f"Default DF file {default_csv_file_path} does not exist. Aborting.")
                return
            else:
                music_data = pd.read_csv(default_csv_file_path, engine='python')
                print(f'Music Data for Visualizer: \n', music_data)
        
        visualizer = AudioDataVisualizer(music_data)
        print("Plotting data...")
        visualizer.visualize(1)
    else:
        print('Skipping Data Visualization')


    # ------------------------------- MusicGenreClassifier
    if train_model:
        prompt_for_gpu()
        print(f"{BLUE}Begin Model Training{RESET}")

        if music_data is None:
            music_data = pd.read_csv(default_csv_file_path)
            
        # Load the data
        dummy_data = pd.read_csv('df_output/dummy_music.csv')
        # Drop non-numeric columns
        X = dummy_data.drop(columns=['filename', 'genre'])
        # Extract target labels
        y = dummy_data['genre']
        # One-hot encode the target labels
        encoder = LabelBinarizer()
        y_encoded = encoder.fit_transform(y)
        # Scale the feature matrix
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        # Define the feature dimension and genres
        feature_dim = X_scaled.shape[1]
        genres = encoder.classes_  # Automatically get genres from encoder
        # Initialize and train the classifier
        classifier = MusicGenreClassifier(feature_dim, genres)
        classifier.train(X_train, y_train, X_test, y_test)
        classifier.evaluate(X_test, y_test)
        
        # classifier = MusicGenreClassifier(dummy_data)
        # X_scaled, y_encoded = classifier.prepare_data()
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        # classifier.train(X_train, y_train, X_test, y_test)
        # classifier.evaluate(X_test, y_test)
    else: print('Skipping Model Training')

    # ------------------------------- Predict a Genre
    # Adjusts input music files dynamically, user input, selections, etc.
    # hard coded single file for now
    if predict_genre:
        print(f"{BLUE}Begin Genre Predictor{RESET}")
        genre = classifier.predict_genre(test_audio_file_1)
        print(f'The predicted genre is: {genre}')
    else: print('Skipping Genre Predictor')


if __name__ == '__main__':
    main()
