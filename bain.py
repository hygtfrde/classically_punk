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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

from CONSTANTS import RED, GREEN, RESET
from helpers import get_input_with_timeout


def convert_string_to_array(value):
    try:
        # print(f"Processing value of type: {type(value)}")
        # print(f"Value starts with: {str(value)[:50]}")
        
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
            # print(f"Column '{col}' data type: {df_input[col].dtype}")
            # print(f"First 10 characters of values in column '{col}':")
            # for value in df_input[col].head(5):
            #     print(f"{str(value)[:50]}")
            # print()
            
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
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
        feature_inputs = X[i]  # Use standard NumPy indexing
        true_label = y[i]  # Use standard NumPy indexing
        # Make prediction
        predicted_class = predict(model, encoder, scaler, feature_inputs)
        # Check if the prediction matches the true label
        if predicted_class == true_label:
            print(f"{GREEN}TRUE: {predicted_class} is {true_label}{RESET}")
            correct_count += 1
        else:
            print(f"{RED}FALSE: {predicted_class} is NOT {true_label}{RESET}")

    # Calculate accuracy
    accuracy = (correct_count / total_count) * 100
    incorrect_count = total_count - correct_count
    
    print(f"Total rows evaluated: {total_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")


def save_encoder_and_scaler(encoder, scaler, encoder_path='pickles/encoder.pkl', scaler_path='pickles/scaler.pkl'):
    try:
        with open(encoder_path, 'wb') as enc_file:
            pickle.dump(encoder, enc_file)
        with open(scaler_path, 'wb') as scal_file:
            pickle.dump(scaler, scal_file)
        print("Encoder and scaler saved successfully.")
    except Exception as e:
        print(f"Error saving encoder and scaler: {e}")
        

def save_data(X_scaled, y, X_scaled_path='pickles/X_scaled.pkl', y_path='pickles/y.pkl'):
    try:
        with open(X_scaled_path, 'wb') as x_file:
            pickle.dump(X_scaled, x_file)
        with open(y_path, 'wb') as y_file:
            pickle.dump(y, y_file)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

    
    
    
    
# --------------------- MAIN
# -----------------------------------------------------------------------------------------
def main(raw_2d_data=True):
    v5_test_5 = 'df_output/v5_5.csv'
    v5_full_stable_test = 'df_output/v5_full.csv'
    v5_reduced_stats = 'df_output/v5_reduced_all_stats.csv'

    if raw_2d_data is True:
        try:
            df_extract = read_raw_str_csv_and_split_df(v5_reduced_stats)
            
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
                loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

                print('Accuracy ======================================')
                # Predict the labels for the test data
                y_pred_probs = model.predict(X_test)
                y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels
                y_test_labels = np.argmax(y_test, axis=1)  # Actual class labels

                # Calculate accuracy
                accuracy = accuracy_score(y_test_labels, y_pred)
                print(f"Test Accuracy: {accuracy:.4f}")

                # Optionally, print the classification report and confusion matrix
                print("Classification Report:")
                print(classification_report(y_test_labels, y_pred))

                print("Confusion Matrix:")
                print(confusion_matrix(y_test_labels, y_pred))
                print('======================================')
            else:
                print("Error: DataFrame is None")
                
            
            # # Save Pickles
            # try:
            #     pickle_dir = 'pickles'
            #     if not os.path.exists(pickle_dir):
            #         os.makedirs(pickle_dir)
            #         print(f"Directory '{pickle_dir}' created.")
            #     else:
            #         print(f"Directory '{pickle_dir}' already exists.")
                
            #     model_path = os.path.join(pickle_dir, 'trained_model.pkl')
                
            #     with open(model_path, 'wb') as model_file:
            #         pickle.dump(model, model_file)
                
            #     save_encoder_and_scaler(encoder, scaler)
            #     save_data(X_scaled, y)
                
            #     print(f"Model saved to '{model_path}'")
            # except Exception as e:
            #     print(f"Error in Pickle: {e}")
            
        except Exception as e:
            print(f"An error occurred in main block: {e}")
    
    
    
    
    
    
    
    else:
        df_extract = pd.read_csv(v5_reduced_stats)
        
        X = df_extract.drop(columns=['filename', 'genre'])
        y = df_extract['genre']
        categories = y.unique()
        num_classes = len(categories)

        # Prepare the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        y_encoded_one_hot = to_categorical(y_encoded, num_classes=num_classes)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded_one_hot, test_size=0.2, random_state=42)

        model, history = build_and_train_model(X_train, y_train, X_test, y_test, X_scaled.shape[1], num_classes)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

        print('Accuracy ======================================')
        # Predict the labels for the test data
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels
        y_test_labels = np.argmax(y_test, axis=1)  # Actual class labels

        # Calculate accuracy
        accuracy = accuracy_score(y_test_labels, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Optionally, print the classification report and confusion matrix
        print("Classification Report:")
        print(classification_report(y_test_labels, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test_labels, y_pred))
        print('======================================')



if __name__ == '__main__':
    main(raw_2d_data=False)