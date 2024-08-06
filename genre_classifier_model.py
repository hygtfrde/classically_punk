import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split

class MusicGenreClassifier:
    def __init__(self, feature_dim, genres):
        self.feature_dim = feature_dim
        self.genres = genres
        self.model = None

    def _build_model(self):
        print('BUILDING MODEL')
        tf.config.set_visible_devices([], 'GPU')  # Force TensorFlow to use CPU

        model = Sequential([
            Input(shape=(self.feature_dim,)),
            Dense(64, activation='relu'),  # Adjust complexity
            Dropout(0.2),
            Dense(len(self.genres), activation='softmax')  # Output layer
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
        print('TRAINING MODEL')
        self.model = self._build_model()
        
        history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_test, y_test),
            verbose=1
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
