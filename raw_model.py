import ast
import sys
import os
import pickle

# TENSORFLOW IS REQUIRED EVEN IF NOT ACCESSED
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

from CONSTANTS import RED, GREEN, RESET


def convert_string_to_array(value):
    try:
        if isinstance(value, str):
            value = value.strip('"').strip("'")
            try:
                value = ast.literal_eval(value)
                
                if isinstance(value, list):
                    # Convert list to numpy array
                    value = np.array(value, dtype=float)
                    # print(f"Converted array shape: {value.shape}")
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
        return None, None
    if df_input is not None:
        for col in df_input.columns:
            if col not in ['filename', 'genre']:
                df_input[col] = df_input[col].apply(convert_string_to_array)
        return df_input
    else:
        print('Error: df_input is None')
        return None, None
    

def prepare_data(X, y):
    try:
        # Step 1: Flatten the features
        X_flattened = X.apply(lambda col: col.apply(lambda x: x.flatten()))
        # Step 2: Convert the DataFrame of flattened arrays into a 2D NumPy array
        X_stacked = np.stack(X_flattened.apply(np.concatenate, axis=1).to_numpy())
        # Step 3: Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_stacked)
        # Step 4: Encode the target labels (y)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        return X_scaled, y_encoded, encoder, scaler
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        return None, None, None, None
    

def build_and_train_model(X_train, y_train, X_test, y_test, num_features, num_classes):
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the callbacks
    history = model.fit(
        X_train, 
        y_train, 
        epochs=3000, 
        batch_size=128, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    return model, history


def predict(model, encoder, scaler, feature_inputs):
    # Scale the feature inputs directly without converting to DataFrame
    feature_inputs_scaled = scaler.transform([feature_inputs])
    # Make predictions
    predictions = model.predict(feature_inputs_scaled)
    # Decode the predictions to category names
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_class


def evaluate_all_rows(model, X, y, encoder, scaler):
    correct_count = 0
    total_count = len(X)
    
    for i in range(total_count):
        # Extract feature inputs and true label
        feature_inputs = X[i]
        true_label = y[i]
        # Make prediction
        predicted_class = predict(model, encoder, scaler, feature_inputs)
        # Check if the prediction matches the true label
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

    
    

# --------------------- MAIN
# -----------------------------------------------------------------------------------------
def main():
    v5_test_5 = 'df_output/v5_5.csv'
    v5_full_stable_test = 'df_output/v5_full.csv'
    v5_reduced_stats = 'df_output/v5_reduced_all_stats.csv'
    v5_kde = 'v5_kde_full_all_stats'

    try:
        df_extract = read_raw_str_csv_and_split_df(v5_full_stable_test)
        
        if df_extract is not None:
            # Split into X and y
            X = df_extract.drop(columns=['filename', 'genre'])
            y = df_extract['genre']
            categories = y.unique()
            num_classes = len(categories)

            # Prepare the data
            X_scaled, y_encoded, encoder, scaler = prepare_data(X, y)
            y_encoded_one_hot = to_categorical(y_encoded, num_classes=num_classes)

            if X_scaled is not None and y_encoded is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded_one_hot, test_size=0.2, random_state=42)
            else:
                print("Error in data preparation")
                raise ValueError("X_scaled or y_encoded is None")
        
            model, history = build_and_train_model(X_train, y_train, X_test, y_test, X_scaled.shape[1], num_classes)
            evaluate_all_rows(model, X_scaled, y, encoder, scaler)
        else:
            print("Error: DataFrame is None")
        
    except Exception as e:
        print(f"An error occurred in main block: {e}")


if __name__ == '__main__':
    main()