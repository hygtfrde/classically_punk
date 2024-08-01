import os
import datetime
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
            print('To manage GPU usage: tensorboard --logdir=logs/fit')
        else:
            print("GPU configuration script not found.")
            

class MusicDataProcessor:
    def __init__(self, dataset_path: str, file_depth_limit: int, excel_output_name: str):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.excel_output_name = excel_output_name
        self.genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        self.data = pd.DataFrame(columns=['filename', 'genre', 'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz'])
    
    def extract_features(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None, None, None, None
        
        print(f'''EXTRACTING: {file_path} \n 
                y: {y} \n
                sr: {sr} \n
                mfcc: {mfcc} \n
                chroma: {chroma} \n
                mel: {mel} \n
                contrast: {contrast} \n
                tonnetz: {tonnetz}
              ''')
        
        return mfcc, chroma, mel, contrast, tonnetz
                
    def load_data(self):
        all_data = []
        for genre in self.genres:
            counter = 0
            genre_dir = os.path.join(self.dataset_path, genre)
            for file in os.listdir(genre_dir):
                if counter >= self.file_depth_limit: break
                file_path = os.path.join(genre_dir, file)
                mfcc, chroma, mel, contrast, tonnetz = self.extract_features(file_path)
                if mfcc is not None:
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        'mfcc': mfcc,
                        'chroma': chroma,
                        'mel': mel,
                        'contrast': contrast,
                        'tonnetz': tonnetz
                    })
                counter += 1
        self.data = pd.DataFrame(all_data)
    
    def get_data(self):
        self.save_data_to_excel(f'{self.excel_output_name}_genres_df.xlsx')
        return self.data

    def save_data_to_excel(self, file_name):
        self.data.to_excel(file_name, index=False)
        print(f'Data saved to {file_name}')



class MusicGenreClassifier:
    def __init__(self, data):
        self.data = data
        self.genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.model = None

    def flatten_features(self, features):
        return np.hstack([np.mean(f, axis=1) for f in features])
    
    def prepare_data(self):
        X = np.array([self.flatten_features([
            row['mfcc'], 
            row['chroma'], 
            row['mel'], 
            row['contrast'], 
            row['tonnetz']
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


# ------------------------------- MAIN -------------------------------
def main():
    prompt_for_gpu()
    
    dataset_path = 'genres'
    genre_classifier = MusicDataProcessor(dataset_path, 3, 'smaller_3_files')
    genre_classifier.load_data()
    music_data = genre_classifier.get_data()
    print('music_data: \n', music_data)

    classifier = MusicGenreClassifier(music_data)
    X_scaled, y_encoded = classifier.prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    classifier.train(X_train, y_train, X_test, y_test)
    classifier.evaluate(X_test, y_test)

    # Predict genre
    # Adjusts input music files dynamically, user input, selections, etc.
    # hard coded sigle file for now
    genre = classifier.predict_genre('genres/blues/blues.00000.wav')
    print(f'The predicted genre is: {genre}')


if __name__ == '__main__':
    main()
