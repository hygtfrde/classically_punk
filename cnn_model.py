import ast
import sys
import os
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from CONSTANTS import RED, GREEN, RESET

def convert_string_to_array(value):
    try:
        if isinstance(value, str):
            value = value.strip('"').strip("'")
            try:
                value = ast.literal_eval(value)
                if isinstance(value, list):
                    value = np.array(value, dtype=float)
                    return value
                else:
                    print("Warning: Evaluated value is not a list.")
            except (ValueError, SyntaxError) as e:
                print(f"Error evaluating string: {e}")
        else:
            print('Value not detected as str')
        
        return value
    except Exception as e:
        print("General failure in conversion:")
        print(f'Error: {e}')
        return value

def read_raw_str_csv_and_split_df(csv_path):
    try:
        df_input = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading csv into df: {e}")
        return None
    if df_input is not None:
        for col in df_input.columns:
            if col not in ['filename', 'genre']:
                df_input[col] = df_input[col].apply(convert_string_to_array)
        return df_input
    else:
        print('Error: df_input is None')
        return None

def prepare_data(X, y):
    try:
        X_flattened = X.apply(lambda col: col.apply(lambda x: x.flatten()))
        X_stacked = np.stack(X_flattened.apply(np.concatenate, axis=1).to_numpy())

        print(f"X_stacked shape: {X_stacked.shape}")  # Debugging line

        num_features = X_stacked.shape[1]
        
        # Choose dimensions close to square
        target_size = 463 * 463
        if num_features < target_size:
            # Pad with zeros if needed
            padding = target_size - num_features
            X_padded = np.pad(X_stacked, ((0, 0), (0, padding)), mode='constant')
        else:
            # Truncate if larger
            X_padded = X_stacked[:, :target_size]

        # Reshape to (num_samples, 463, 463, 1) for CNN
        X_reshaped = X_padded.reshape(-1, 463, 463, 1)
        X_scaled = X_reshaped

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_encoded_one_hot = to_categorical(y_encoded, num_classes=len(encoder.classes_))

        return X_scaled, y_encoded_one_hot, encoder, None  # Return scaler as None if not used
        
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        return None, None, None, None




def build_and_train_model(X_train, y_train, X_test, y_test, input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, 
        y_train, 
        epochs=1000,
        batch_size=128, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return model, history

def predict(model, encoder, feature_inputs):
    # No scaling is needed for CNN input
    feature_inputs_reshaped = feature_inputs.reshape(1, *feature_inputs.shape)
    predictions = model.predict(feature_inputs_reshaped)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_class

def evaluate_all_rows(model, X, y, encoder):
    correct_count = 0
    total_count = len(X)
    
    for i in range(total_count):
        feature_inputs = X[i]  # Use standard NumPy indexing
        true_label_index = np.argmax(y[i])  # Convert one-hot to index
        true_label = encoder.inverse_transform([true_label_index])[0]  # Convert index to class name
        predicted_class = predict(model, encoder, feature_inputs)
        if predicted_class == true_label:
            print(f"{GREEN}TRUE: {predicted_class} is {true_label}{RESET}")
            correct_count += 1
        else:
            print(f"{RED}FALSE: {predicted_class} is NOT {true_label}{RESET}")

    accuracy = (correct_count / total_count) * 100
    incorrect_count = total_count - correct_count
    
    print(f"Total rows evaluated: {total_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")




def main():
    v5_test_5 = 'df_output/v5_5.csv'
    v5_full = 'df_output/v5_full.csv'
    
    try:
        df_extract = read_raw_str_csv_and_split_df(v5_full)
        
        if df_extract is not None:
            X = df_extract[['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz']]
            y = df_extract['genre']
            categories = y.unique()
            num_classes = len(categories)

            X_scaled, y_encoded, encoder, _ = prepare_data(X, y)  # Use _ for scaler as it's not used
            if X_scaled is not None and y_encoded is not None:
                print(f"Prepared data shape: X_scaled {X_scaled.shape}, y_encoded {y_encoded.shape}")  # Debugging line

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

                # No need to reshape again, X_scaled is already in the correct shape
                model, history = build_and_train_model(X_train, y_train, X_test, y_test, (463, 463, 1), num_classes)
                evaluate_all_rows(model, X_test, y_test, encoder)

            else:
                print("Error in data preparation")
                raise ValueError("X_scaled or y_encoded is None")
        
        else:
            print("Error: DataFrame is None")
        
    except Exception as e:
        print(f"An error occurred in main block: {e}")

if __name__ == '__main__':
    main()
