import os
import datetime
import ast
import librosa
import pandas as pd
import openpyxl
import importlib.util
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard



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
            

class MusicDataProcessor:
    def __init__(self, dataset_path: str, file_depth_limit: int, excel_output_name: str):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.excel_output_name = excel_output_name
        self.genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        self.data = pd.DataFrame(columns=['filename', 'genre', 'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz', 'harmony', 'perceptr', 'tempo'])
    
    def extract_features(self, file_path):
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
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
        
        print(f'''EXTRACTING: {file_path} \n 
                y: {y} \n
                sr: {sr} \n
                mfcc: {mfcc} \n
                chroma: {chroma} \n
                mel: {mel} \n
                contrast: {contrast} \n
                tonnetz: {tonnetz}
                harmony: {harmony} \n
                perceptr: {perceptr} \n
                tempo: {tempo}
              ''')
        
        return mfcc, chroma, mel, contrast, tonnetz, harmony, perceptr, tempo
                
    def load_data(self):
        all_data = []
        for genre in self.genres:
            counter = 0
            genre_dir = os.path.join(self.dataset_path, genre)
            for file in os.listdir(genre_dir):
                if counter >= self.file_depth_limit: break
                file_path = os.path.join(genre_dir, file)
                mfcc, chroma, mel, contrast, tonnetz, harmony, perceptr, tempo = self.extract_features(file_path)
                if mfcc is not None:
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        'mfcc': mfcc,
                        'chroma': chroma,
                        'mel': mel,
                        'contrast': contrast,
                        'tonnetz': tonnetz,
                        'harmony': harmony,
                        'perceptr': perceptr,
                        'tempo': tempo
                    })
                counter += 1
        self.data = pd.DataFrame(all_data)
    
    def save_data_to_excel(self, file_name):
        self.data.to_excel(file_name, index=False)
        print(f'Data saved to {file_name}')

    def get_data(self):
        df_output_dir = 'df_output'
        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")
        self.save_data_to_excel(f'{df_output_dir}/{self.excel_output_name}_genres_df.xlsx')
        return self.data




class MusicGenreClassifier:
    def __init__(self, data):
        self.data = data
        self.genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.model = None

    
    def flatten_features(self, features):
        flattened_features = []
        
        for f in features:
            if isinstance(f, str):
                try:
                    # Convert string representation of array back to numpy array
                    f = np.array(ast.literal_eval(f))
                except (SyntaxError, ValueError) as e:
                    print(f"Error converting string to array: {e}")
                    continue

            if f is None or not hasattr(f, 'size') or not f.size:
                print(f"Warning: Skipping empty or None feature: {f}")
                continue
            
            print(f"Feature shape before flattening: {f.shape}")
            
            if len(f.shape) == 1:
                # If the feature array is 1D, simply append it as is
                flattened_features.append(f)
            elif len(f.shape) == 2 and f.shape[1] > 1:
                # Compute mean along axis 1 if the feature array is 2D
                flattened_features.append(np.mean(f, axis=1))
            else:
                print(f"Warning: Unexpected feature shape: {f.shape}. Skipping this feature.")
        
        if not flattened_features:
            print("Error: No valid features to flatten.")
            return np.array([])
        
        # Stack the flattened features horizontally
        return np.hstack(flattened_features)

    
    def prepare_data(self):
        X = np.array([self.flatten_features([
            row['mfcc'], 
            row['chroma'], 
            row['mel'], 
            row['contrast'], 
            row['tonnetz'],
            row['harmony'],
            row['perceptr'],
            row['tempo']
        ]) for _, row in self.data.iterrows()])

        print(f"Feature shape before scaling: {X.shape}")

        y = self.data['genre'].values
        y_encoded = self.encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
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



def get_user_input(prompt, default_value=True):
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'n']:
            return response == 'y'
        elif response == '':
            return default_value
        else:
            print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")


# ------------------------------- MAIN -------------------------------
def main():
    BLUE = '\033[34m'
    RESET = '\033[0m'
    
    print("Configure the following options:")
    process_data = get_user_input("Do you want to process data? (Y/N): ", default_value=True)
    train_model = get_user_input("Do you want to train the model? (Y/N): ", default_value=True)
    predict_genre = get_user_input("Do you want to predict genre? (Y/N): ", default_value=True)
    
    
    # ------------------------------- MusicDataProcessor
    music_data = None
    if process_data:
        print(f"{BLUE}Begin Data Processing{RESET}")
        dataset_path = 'genres'
        genre_classifier = MusicDataProcessor(dataset_path, 3, 'smaller_3_files')
        genre_classifier.load_data()
        music_data = genre_classifier.get_data()
        print('music_data: \n', music_data)
    else: print('Skipping Data Processing')


    # ------------------------------- MusicGenreClassifier
    if train_model:
        prompt_for_gpu()
        print(f"{BLUE}Begin Model Training{RESET}")
        
        static_df = pd.read_excel('df_output/smaller_3_files_genres_df.xlsx')
        print(static_df['mfcc'].head())
        print(static_df['chroma'].head())
        
        classifier = MusicGenreClassifier(static_df)
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
        genre = classifier.predict_genre('genres/blues/blues.00000.wav')
        print(f'The predicted genre is: {genre}')
    else: print('Skipping Genre Predictor')


if __name__ == '__main__':
    main()
