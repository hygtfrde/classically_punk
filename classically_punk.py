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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

"""
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

genres_from_dataset = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
columns_from_extracted_librosa_audio_data = ['filename', 'genre', 'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz', 'harmony_mean', 'harmony_std', 'perceptr_mean', 'perceptr_std', 'tempo']
test_audio_file_1 = 'genres/blues/blues.00000.wav'
BLUE = '\033[34m'
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'



class AudioDataVisualizer:
    def __init__(self, df):
        self.df = df
        self.ensure_directories()

    def ensure_directories(self):
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        if not os.path.exists('visualizations/_test_audio_waveform_spectogram'):
            os.makedirs('visualizations/_test_audio_waveform_spectogram')

    # ------------------------------- AUDIO FILE VISUALIZERS
    def plot_waveform(self, audio_data, sample_rate, filename):
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
        plt.title(f'Waveform - {filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig(f'visualizations/_test_audio_waveform_spectogram/{filename}_waveform.png')
        plt.close()

    def plot_spectrogram(self, audio_data, sample_rate, filename):
        frequencies, times, Sxx = spectrogram(audio_data, sample_rate)
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
        plt.title(f'Spectrogram - {filename}')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Intensity [dB]')
        plt.savefig(f'visualizations/_test_audio_waveform_spectogram/{filename}_spectrogram.png')
        plt.close()
    # -------------------------------

    # ------------------------------- LIBROSA AUDIO DATA VISUALIZERS
    def plot_mfccs(self, mfccs, filename):
        # Ensure mfccs is a numerical NumPy array
        mfccs = np.array(mfccs, dtype=np.float32)  # Convert to float32 for numerical operations

        # Handle potential non-finite values
        mfccs = np.nan_to_num(mfccs, copy=False)  # Replace NaN and inf with 0

        # Ensure all values in mfccs are finite
        mfccs = np.where(np.isfinite(mfccs), mfccs, 0)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCC - {filename}')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.savefig(f'visualizations/{filename}_mfccs.png')
        plt.close()
        
    def plot_chroma(self, chroma, filename):
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(chroma, x_axis='time')
        plt.colorbar()
        plt.title(f'Chroma - {filename}')
        plt.xlabel('Time')
        plt.ylabel('Chroma Coefficients')
        plt.savefig(f'visualizations/{filename}_chroma.png')
        plt.close()
        
    def plot_feature(self, feature, feature_name, filename):
        plt.figure(figsize=(12, 6))

        if feature_name in ['Chroma', 'Mel', 'Contrast', 'Tonnetz']:
            # Assuming these features are 2D
            if feature.ndim == 2:
                librosa.display.specshow(feature, x_axis='time')
                plt.colorbar()
            else:
                raise ValueError(f'Feature for {feature_name} must be a 2D array.')
        elif feature_name in ['Harmony', 'Perceptr']:
            # Assuming these features are 1D
            print(f'{feature_name} => feature.ndim: {feature.ndim}')
            if feature.ndim == 1:
                self.plot_1d_feature(feature, feature_name, filename)
            else:
                raise ValueError(f'Feature for {feature_name} must be a 1D array.')
        else:
            raise ValueError(f'Unsupported feature type: {feature_name}')
        
        plt.savefig(f'visualizations/{filename}_{feature_name.lower()}.png')
        plt.close()

    def plot_1d_feature(self, feature, feature_name, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(feature)
        plt.title(f'{feature_name} - {filename}')
        plt.xlabel('Time')
        plt.ylabel(f'{feature_name} Values')
        plt.savefig(f'visualizations/{filename}_{feature_name.lower()}.png')
        plt.close()

    def plot_scalar_feature(self, value, feature_name, filename):
        plt.figure(figsize=(8, 4))
        plt.bar(feature_name, value)
        plt.title(f'{feature_name} - {filename}')
        plt.ylabel(f'{feature_name} Value')
        plt.savefig(f'visualizations/{filename}_{feature_name.lower()}.png')
        plt.close()
    # -------------------------------

    def visualize(self, rows_visualized_per_genre):
        # Plot Test Audio File Waveform and Spectogram
        audio_data, sample_rate = librosa.load(test_audio_file_1, sr=None)
        self.plot_waveform(audio_data, sample_rate, 'test_blues_00000')
        self.plot_spectrogram(audio_data, sample_rate, 'test_blues_00000')
        
        # Plots for each genre limited by rows_visualized_per_genre
        visualized_counts = {}
        for idx, row in self.df.iterrows():
            genre = row['genre']
            if visualized_counts.get(genre, 0) >= rows_visualized_per_genre:
                continue
            filename = row['filename']

            # Extract features from the DataFrame
            mfccs = row['mfcc']
            chroma = row['chroma']
            mel = row['mel']
            contrast = row['contrast']
            tonnetz = row['tonnetz']
            harmony_mean = row['harmony_mean']
            harmony_std = row['harmony_std']
            perceptr_mean = row['perceptr_mean']
            perceptr_std = row['perceptr_std']
            tempo = row['tempo']

            # Convert features to numpy arrays if necessary for plotting
            mfccs = np.array(mfccs)
            chroma = np.array(chroma)
            mel = np.array(mel)
            contrast = np.array(contrast)
            tonnetz = np.array(tonnetz)

            # Plot and save visualizations
            self.plot_mfccs(mfccs, filename)
            self.plot_feature(chroma, 'Chroma', filename)
            self.plot_feature(mel, 'Mel', filename)
            self.plot_feature(contrast, 'Contrast', filename)
            self.plot_feature(tonnetz, 'Tonnetz', filename)
            
            # Plot scalar features
            self.plot_scalar_feature(harmony_mean, 'Harmony Mean', filename)
            self.plot_scalar_feature(harmony_std, 'Harmony Std', filename)
            self.plot_scalar_feature(perceptr_mean, 'Perceptr Mean', filename)
            self.plot_scalar_feature(perceptr_std, 'Perceptr Std', filename)
            self.plot_scalar_feature(tempo, 'Tempo', filename)

            visualized_counts[genre] = visualized_counts.get(genre, 0) + 1
            
            print(f'Visualizations for {filename} saved.')

            # Stop if the limit is reached for this genre
            if visualized_counts[genre] >= rows_visualized_per_genre:
                print(f'Reached limit of {rows_visualized_per_genre} visualizations for genre: {genre}')
                continue




class MusicDataProcessor:
    def __init__(self, dataset_path: str, file_depth_limit: int, excel_output_name: str):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.excel_output_name = excel_output_name
        self.genres = genres_from_dataset
        self.data = pd.DataFrame(columns=columns_from_extracted_librosa_audio_data)

    def extract_features(self, file_path, verbose=1):
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            harmony = librosa.effects.harmonic(y)
            perceptr = librosa.effects.percussive(y)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Extract Mean and StdDev from harmony and perceptr
            harmony_mean = np.mean(harmony)
            harmony_std = np.std(harmony)
            perceptr_mean = np.mean(perceptr)
            perceptr_std = np.std(perceptr)

            # Debug output with increased precision
            # np.set_printoptions(precision=10, suppress=True)  # Set precision and suppress scientific notation
            print(f"Harmony: Min={np.min(harmony):.10f}, Max={np.max(harmony):.10f}, Range={np.ptp(harmony):.10f}")
            print(f"Perciptr: Min={np.min(perceptr):.10f}, Max={np.max(perceptr):.10f}, Range={np.ptp(perceptr):.10f}")

            if verbose == 1:
                print(f"EXTRACTING: {file_path} \n"
                    f"  y: {y.shape} \n"
                    f"  sr: {sr} \n"
                    f"  mfcc: {mfcc.shape} \n"
                    f"  chroma: {chroma.shape} \n"
                    f"  mel: {mel.shape} \n"
                    f"  contrast: {contrast.shape} \n"
                    f"  tonnetz: {tonnetz.shape} \n"
                    f"  harmony_mean: {harmony_mean:.10f} \n"
                    f"  harmony_std: {harmony_std:.10f} \n"
                    f"  perceptr_mean: {perceptr_mean:.10f} \n"
                    f"  perceptr_std: {perceptr_std:.10f} \n"
                    f"  tempo: {tempo}")

            return mfcc, chroma, mel, contrast, tonnetz, harmony_mean, harmony_std, perceptr_mean, perceptr_std, tempo

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None


    def validate_data(self):
        """Validate the data to ensure no missing values and proper data types."""
        valid = True
        # VALIDATE NO 
        if self.data.isnull().values.any():
            valid = False
            raise ValueError("Data contains missing values.")
        if not all(self.data['genre'].isin(self.genres)):
            valid = False
            raise ValueError("Data contains invalid genres.")
        # Additional checks ......
        return valid
        
    def load_data(self):
        all_data = []
        for genre in self.genres:
            counter = 0
            genre_dir = os.path.join(self.dataset_path, genre)
            for file in os.listdir(genre_dir):
                if counter >= self.file_depth_limit:
                    break
                file_path = os.path.join(genre_dir, file)
                features = self.extract_features(file_path, 0)
                if features is not None:
                    mfcc, chroma, mel, contrast, tonnetz, harmony_mean, harmony_std, perceptr_mean, perceptr_std, tempo = features
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        'mfcc': mfcc.tolist(),  # Convert arrays to lists if needed
                        'chroma': chroma.tolist(),
                        'mel': mel.tolist(),
                        'contrast': contrast.tolist(),
                        'tonnetz': tonnetz.tolist(),
                        'harmony_mean': harmony_mean,  # Store mean and std as separate values
                        'harmony_std': harmony_std,
                        'perceptr_mean': perceptr_mean,
                        'perceptr_std': perceptr_std,
                        'tempo': tempo
                    })
                counter += 1

        self.data = pd.DataFrame(all_data)
        validate_data_after_loading = self.validate_data()
        print('Is data validated after loading? => ', validate_data_after_loading)

    def get_data(self):
        df_output_dir = 'df_output'
        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")
        
        file_path = os.path.join(df_output_dir, f'{self.excel_output_name}_genres_df.xlsx')
        
        self.data.to_excel(file_path, index=False)
        print(f"Data saved to {file_path}")

        return self.data



class MusicGenreClassifier:
    def __init__(self, data):
        self.data = data
        self.genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.model = None


    def preprocess_string(self, s):
        # Replace spaces with commas
        s = s.replace(' ', ',')
        # Add necessary brackets if missing
        if not (s.startswith('[') and s.endswith(']')):
            s = f'[{s}]'
        return s

    def parse_feature(self, feature_str):
        try:
            processed_str = self.preprocess_string(feature_str)
            feature_list = json.loads(processed_str)
            return np.array(feature_list)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error converting string to array: {e}")
            print(f"Invalid string: {feature_str}")
            return None


    def validate_dataframe(self):
        valid = True
        # Check for non-null and correct data types
        for column in ['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz', 'harmony_mean', 'harmony_std', 'perceptr_mean', 'perceptr_std', 'tempo']:
            if self.data[column].isnull().any():
                print(f"Error: Column '{column}' contains null values.")
                valid = False

            # Validate each cell in the feature columns
            for i, feature in enumerate(self.data[column]):
                parsed_feature = self.parse_feature(feature)
                if parsed_feature is None:
                    print(f"Error: Failed to parse feature at row {i} in column '{column}'.")
                    valid = False
                elif len(parsed_feature.shape) != 1:
                    print(f"Error: Feature at row {i} in column '{column}' is not 1D. Shape: {parsed_feature.shape}")
                    valid = False
        print(f"Is DF validated? ===> {valid}")
        return valid



    def flatten_features(self, features):
        flattened_features = []

        for f in features:
            if isinstance(f, str):
                print(f"Original string: {f}")
                try:
                    # Convert string representation of array back to numpy array
                    f = np.array(json.loads(f))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error converting string to array: {e}")
                    print(f"Invalid string: {f}")
                    continue

            if f is None or not hasattr(f, 'size') or not f.size:
                print(f"Warning: Skipping empty or None feature: {f}")
                continue

            print(f"Feature shape before flattening: {f.shape}")

            if len(f.shape) == 1:
                flattened_features.append(f)
            elif len(f.shape) == 2 and f.shape[1] > 1:
                flattened_features.append(np.mean(f, axis=1))
            else:
                print(f"Warning: Unexpected feature shape: {f.shape}. Skipping this feature.")

        if not flattened_features:
            print("Error: No valid features to flatten.")
            return np.array([])

        return np.hstack(flattened_features)

    
    def prepare_data(self):
        # Ensure flatten_features can handle the new feature types
        def flatten_features(features):
            flat_features = []
            for feature in features:
                if isinstance(feature, (list, np.ndarray)):
                    flat_features.extend(np.ravel(feature))  # Flatten arrays
                else:
                    flat_features.append(feature)  # Add scalars directly
            return flat_features

        # Prepare the feature matrix
        X = np.array([self.flatten_features([
            row['mfcc'], 
            row['chroma'], 
            row['mel'], 
            row['contrast'], 
            row['tonnetz'],
            [row['harmony_mean'], row['harmony_std']],  # Flatten mean and std for harmony
            [row['perceptr_mean'], row['perceptr_std']],  # Flatten mean and std for perceptr
            row['tempo']
        ]) for _, row in self.data.iterrows()])

        print(f"Feature shape before scaling: {X.shape}")

        # Encode target labels
        y = self.data['genre'].values
        y_encoded = self.encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Update feature dimension
        self.feature_dim = X_scaled.shape[1]
        
        return X_scaled, y_encoded


    def _build_model(self):
        if self.feature_dim is None:
            raise ValueError("Feature dimension is not set. Call prepare_data() first.")

        model = Sequential([
            Input(shape=(self.feature_dim,)),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(self.genres), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        self.model = self._build_model()
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tensorboard_callback]
        )
        
        print("Training history:")
        for key in history.history.keys():
            print(f"{key}: {history.history[key]}")

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model is not built. Call train() first.")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test accuracy: {accuracy}')
        return accuracy
    
    def predict_genre(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        features = self.flatten_features([mfcc, chroma, mel, contrast, tonnetz])
        features_scaled = self.scaler.transform([features])
        
        prediction = self.model.predict(features_scaled)
        genre = self.encoder.inverse_transform([np.argmax(prediction)])
        
        return genre[0]




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
    process_data = get_user_input("Do you want to process data? (Y/N): ", default_value=True)
    visualize_data = get_user_input("Do you want to visualize data? (Y/N): ", default_value=True)
    train_model = get_user_input("Do you want to train the model? (Y/N): ", default_value=True)
    predict_genre = get_user_input("Do you want to predict genre? (Y/N): ", default_value=True)
    
    
    # ------------------------------- MusicDataProcessor
    music_data = None
    if process_data:
        print(f"{BLUE}Begin Data Processing{RESET}")
        dataset_path = 'genres'
        genre_classifier = MusicDataProcessor(dataset_path, 1, 'just_1_file')

        print("Loading data...")
        genre_classifier.load_data()
        print("Data loaded successfully and validated.")

        print("Getting data...")
        music_data = genre_classifier.get_data()
        print('Music data: \n', music_data)
    else: print('Skipping Data Processing')
    
    
    # ------------------------------- AudioDataVisualizer
    default_df_file_path = 'df_output/_default.xlsx'
    if visualize_data:
        if music_data is None:
            if not os.path.exists(default_df_file_path):
                print(f"Default DF file {default_df_file_path} does not exist. Aborting.")
                return
            else:
                music_data = pd.read_excel(default_df_file_path)
        
        visualizer = AudioDataVisualizer(music_data)
        visualizer.visualize(1)
    else:
        print('Skipping Data Visualization')


    # ------------------------------- MusicGenreClassifier
    if train_model:
        prompt_for_gpu()
        print(f"{BLUE}Begin Model Training{RESET}")

        static_df = pd.read_excel('df_output/just_1_file_genres_df.xlsx')
        print(static_df['mfcc'].head())
        print(static_df['chroma'].head())
        
        classifier = MusicGenreClassifier(static_df)
        classifier.validate_dataframe()
        if not classifier: 
            print(f'{RED}validate_dataframe failed{RESET}')
            return
        else: (f'{GREEN}validate_dataframe success{RESET}')
        
        X_scaled, y_encoded = classifier.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        classifier.train(X_train, y_train, X_test, y_test)
        classifier.evaluate(X_test, y_test)
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
