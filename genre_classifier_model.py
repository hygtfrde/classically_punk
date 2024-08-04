import json
import datetime

import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import TensorBoard


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
