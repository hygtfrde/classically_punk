import json
import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import TensorBoard
import librosa



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
                # Convert string representation of array back to numpy array
                f = np.array(json.loads(f))
            if f is None or not hasattr(f, 'size') or not f.size:
                continue
            if len(f.shape) == 1:
                flattened_features.append(f)
            elif len(f.shape) == 2 and f.shape[1] > 1:
                flattened_features.append(np.mean(f, axis=1))
            else:
                flattened_features.append(f.flatten())
        if not flattened_features:
            return np.array([])
        return np.hstack(flattened_features)

    def prepare_data(self):
        def flatten_features(features):
            flat_features = []
            for feature in features:
                if isinstance(feature, str):
                    # Convert string representation of array back to list
                    feature = json.loads(feature)
                if isinstance(feature, (list, np.ndarray)):
                    flat_features.extend(np.ravel(feature))  # Flatten arrays
                else:
                    flat_features.append(feature)  # Add scalars directly
            return flat_features

        # Prepare the feature matrix
        X = np.array([flatten_features([
            row['mfcc_mean'], row['mfcc_std'],
            row['chroma_mean'], row['chroma_std'],
            row['mel_mean'], row['mel_std'],
            row['contrast_mean'], row['contrast_std'],
            row['tonnetz_mean'], row['tonnetz_std'],
            row['harmony_mean'], row['harmony_std'],
            row['perceptr_mean'], row['perceptr_std'],
            row['tempo']
        ]) for _, row in self.data.iterrows()])

        # Ensure all features are numerical
        X = np.array(X, dtype=np.float32)
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
        features = self.extract_features(audio_path)
        
        if features is None:
            return None
        
        flattened_features = self.flatten_features([
            features['mfcc_mean'], 
            features['mfcc_std'], 
            features['chroma_mean'], 
            features['chroma_std'], 
            features['mel_mean'], 
            features['mel_std'], 
            features['contrast_mean'], 
            features['contrast_std'], 
            features['tonnetz_mean'], 
            features['tonnetz_std'],
            features['harmony_mean'], 
            features['harmony_std'], 
            features['perceptr_mean'], 
            features['perceptr_std'], 
            features['tempo']
        ])
        
        features_scaled = self.scaler.transform([flattened_features])
        
        prediction = self.model.predict(features_scaled)
        genre = self.encoder.inverse_transform([np.argmax(prediction)])
        
        return genre[0]

    def extract_features(self, file_path, verbose='v'):
        try:
            y, sr = librosa.load(file_path, sr=None)
            start_time = 0
            end_time = 30  # Extract first 30 seconds

            # Convert start and end times to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Ensure end_sample is within the length of y
            end_sample = min(end_sample, len(y))

            # Slice audio snippet
            y = y[start_sample:end_sample]
            
            # Handle cases where y is too short
            n_fft = 1024
            if len(y) < n_fft:
                n_fft = len(y)  # Adjust n_fft to the length of y

            # Extract features from the snippet
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            harmony = librosa.effects.harmonic(y)
            perceptr = librosa.effects.percussive(y)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Extract Mean and StdDev for all features
            mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
            mfcc_std = np.std(mfcc, axis=1).astype(np.float32)
            chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
            chroma_std = np.std(chroma, axis=1).astype(np.float32)
            mel_mean = np.mean(mel, axis=1).astype(np.float32)
            mel_std = np.std(mel, axis=1).astype(np.float32)
            contrast_mean = np.mean(contrast, axis=1).astype(np.float32)
            contrast_std = np.std(contrast, axis=1).astype(np.float32)
            tonnetz_mean = np.mean(tonnetz, axis=1).astype(np.float32)
            tonnetz_std = np.std(tonnetz, axis=1).astype(np.float32)
            harmony_mean = np.mean(harmony).astype(np.float32)
            harmony_std = np.std(harmony).astype(np.float32)
            perceptr_mean = np.mean(perceptr).astype(np.float32)
            perceptr_std = np.std(perceptr).astype(np.float32)

            if verbose == 'v':
                print(f"EXTRACTING: {file_path} \n"
                    f"  y: {y.shape}, dtype: {y.dtype} \n"
                    f"  sr: {sr} \n"
                    f"  mfcc_mean: {mfcc_mean.shape}, dtype: {mfcc_mean.dtype} \n"
                    f"  chroma_mean: {chroma_mean.shape}, dtype: {chroma_mean.dtype} \n"
                    f"  mel_mean: {mel_mean.shape}, dtype: {mel_mean.dtype} \n"
                    f"  contrast_mean: {contrast_mean.shape}, dtype: {contrast_mean.dtype} \n"
                    f"  tonnetz_mean: {tonnetz_mean.shape}, dtype: {tonnetz_mean.dtype} \n"
                    f"  harmony_mean: {harmony_mean:.10f}, dtype: {type(harmony_mean)} \n"
                    f"  harmony_std: {harmony_std:.10f}, dtype: {type(harmony_std)} \n"
                    f"  perceptr_mean: {perceptr_mean:.10f}, dtype: {type(perceptr_mean)} \n"
                    f"  perceptr_std: {perceptr_std:.10f}, dtype: {type(perceptr_std)} \n"
                    f"  tempo: {tempo}, dtype: {type(tempo)}")

            return {
                'mfcc': mfcc,
                'chroma': chroma,
                'mel': mel,
                'contrast': contrast,
                'tonnetz': tonnetz,
                'harmony_mean': harmony_mean,
                'harmony_std': harmony_std,
                'perceptr_mean': perceptr_mean,
                'perceptr_std': perceptr_std,
                'tempo': tempo,
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'chroma_mean': chroma_mean,
                'chroma_std': chroma_std,
                'mel_mean': mel_mean,
                'mel_std': mel_std,
                'contrast_mean': contrast_mean,
                'contrast_std': contrast_std,
                'tonnetz_mean': tonnetz_mean,
                'tonnetz_std': tonnetz_std
            }

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
