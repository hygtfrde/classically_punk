import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

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
    dummy_path = 'df_output/dummy_music.csv' 
    moment_of_truth_path = 'df_output/test.csv'
    
    X, y = read_csv_and_prepare_data(moment_of_truth_path)
    
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

if __name__ == "__main__":
    main()
