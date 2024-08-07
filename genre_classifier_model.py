import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class MusicGenreClassifier:
    def __init__(self):
        self.model = None
        self.df = None

    def read_direct_df(self, input_df):        
        print('DataFrame Head:\n', input_df.head())
        print('DataFrame Info:\n', input_df.info())
        
        X = input_df.drop(columns=['filename', 'genre'])
        y = input_df['genre']
        
        return X, y

    def read_csv_and_prepare_data(self, file_path):
        self.df = pd.read_csv(file_path)
        
        print('DataFrame Head:\n', self.df.head())
        print('DataFrame Info:\n', self.df.info())
        
        X = self.df.drop(columns=['filename', 'genre'])
        y = self.df['genre']
        
        return X, y

    def prepare_data(self, X, y, categories):
        encoder = OneHotEncoder(sparse_output=False, categories=[categories])
        y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y_encoded, encoder

    def build_and_train_model(self, X_train, y_train, X_test, y_test, num_features, num_classes):
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
