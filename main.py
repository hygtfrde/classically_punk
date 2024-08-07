import os
import datetime
import ast
import json
import importlib.util

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

from audio_vizualizer import AudioDataVisualizer
from music_processor import MusicDataProcessor
from genre_classifier_model import MusicGenreClassifier



# ------------------------------- GLOBAL VARS
test_audio_file_1 = 'genres/blues/blues.00000.wav'
dataset_path = 'genres'
default_csv_file_path = 'df_output/test_1.csv'

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
def full_etl_process():
    music_data = None
    print("Configure the following options:")
    process_data = get_user_input("Do you want to process the data? (Y/N): ", default_value=True)
    visualize_data = get_user_input("Do you want to visualize data? (Y/N): ", default_value=True)
    train_model = get_user_input("Do you want to train the model? (Y/N): ", default_value=True)
    predict_genre = get_user_input("Do you want to predict genre? (Y/N): ", default_value=True)
    
    
    # ------------------------------- MusicDataProcessor
    if process_data:
        print(f"{BLUE}Begin Data Processing{RESET}")
        genre_classifier = MusicDataProcessor(dataset_path, None, 'test_1')

        print("Loading data...")
        genre_classifier.load_data()
        print(f"{GREEN}Data loaded successfully!{RESET}")

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
        # Plot Test Audio File Waveform and Spectogram of Test Audio
        audio_data, sample_rate = librosa.load(test_audio_file_1, sr=None)
        visualizer.plot_waveform(audio_data, sample_rate, 'test_blues_00000')
        visualizer.plot_spectrogram(audio_data, sample_rate, 'test_blues_00000')
        visualizer.visualize(1)
    else:
        print('Skipping Data Visualization')


    # ------------------------------- MusicGenreClassifier
    if train_model:
        prompt_for_gpu()
        print(f"{BLUE}Begin Model Training{RESET}")

    else: print('Skipping Model Training')

    # ------------------------------- Predict a Genre
    # Adjusts input music files dynamically, user input, selections, etc.
    # hard coded single file for now
    # if predict_genre:
    #     print(f"{BLUE}Begin Genre Predictor{RESET}")
    #     genre = classifier.predict_genre(test_audio_file_1)
    #     print(f'The predicted genre is: {genre}')
    # else: print('Skipping Genre Predictor')
    

# Disable all GPU devices
tf.config.set_visible_devices([], 'GPU')

def read_csv_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    print('DataFrame Head:\n', df.head())
    print('DataFrame Info:\n', df.info())
    
    X = df.drop(columns=['filename', 'genre'])
    y = df['genre']
    
    return X, y

def prepare_data(X, y, categories):
    encoder = OneHotEncoder(sparse_output=False, categories=[categories])
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, encoder

def build_and_train_model(X_train, y_train, X_test, y_test, num_features, num_classes):
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, 
        y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return history



def main():
    test_csv_path = 'df_output/test_1.csv'

    try:
        X, y = read_csv_and_prepare_data(test_csv_path)

        categories = y.unique()
        num_classes = len(categories)

        X_scaled, y_encoded, encoder = prepare_data(X, y, categories)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        print(f"Training feature matrix shape: {X_train.shape}")
        print(f"Testing feature matrix shape: {X_test.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Testing target shape: {y_test.shape}")

        history = build_and_train_model(X_train, y_train, X_test, y_test, X_scaled.shape[1], num_classes)

        print("Training history:")
        for key in history.history.keys():
            print(f"{key}: {history.history[key]}")
        
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    main()