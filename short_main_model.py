import ast
import threading
import queue
import sys
import os
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical


GREEN = '\033[32m'
RED = '\033[31m'
RESET = '\033[0m'



def convert_string_to_array(value):
    try:
        print(f"Processing value of type: {type(value)}")
        print(f"Value starts with: {str(value)[:50]}")
        
        if isinstance(value, str):
            value = value.strip('"').strip("'")
            try:
                value = ast.literal_eval(value)
                
                if isinstance(value, list):
                    # Convert list to numpy array
                    value = np.array(value, dtype=float)
                    print(f"Converted array shape: {value.shape}")
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
            print(f"Column '{col}' data type: {df_input[col].dtype}")
            print(f"First 10 characters of values in column '{col}':")
            for value in df_input[col].head(5):
                print(f"{str(value)[:50]}")
            print()
            
            if col not in ['filename', 'genre', 'harmony', 'perceptr', 'tempo']:
                df_input[col] = df_input[col].apply(convert_string_to_array)
        
        print('BEGIN SHAPE TEST ----------------------------------------------------- ')
        X = df_input.drop(columns=['filename', 'genre', 'harmony', 'perceptr', 'tempo'])
        for col in X.columns:
            print(f"Column '{col}', dtype: {X[col].dtype}")
            for value in X[col].head(5):
                print(f"Value type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")
        print('END SHAPE TEST ----------------------------------------------------- ')

        return df_input
    else:
        print('Error: df_input is None')
        return None, None
    


def pad_or_truncate(array, target_shape):
    if array.shape[1] < target_shape[1]:
        padding = ((0, 0), (0, target_shape[1] - array.shape[1]))
        return np.pad(array, padding, mode='constant')
    elif array.shape[1] > target_shape[1]:
        return array[:, :target_shape[1]]
    else:
        return array


def process_data(X, y, target_shape):
    # Ensure X is numeric and consistent in shape
    X_numeric = X.apply(lambda col: np.array([np.array(x) for x in col]))
    X_padded = X_numeric.apply(lambda col: pad_or_truncate(np.array(col.tolist()), target_shape))
    
    # Flatten if necessary
    X_flattened = X_padded.apply(lambda col: np.array([x.flatten() for x in col]))

    # Convert to NumPy array
    X_array = np.array(X_flattened.tolist())
    
    # One-hot encode y
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    
    return X_array, y_encoded



def normalize_array_lengths(X, target_length):
    return X.map(lambda x: pad_or_truncate(x, target_length))

def find_min_length(X):
    # Check that this function is returning an integer
    lengths = [arr.shape[1] for arr in X if isinstance(arr, np.ndarray)]
    min_length = min(lengths) if lengths else 0
    print(f"find_min_length - lengths: {lengths}, min_length: {min_length}")
    return min_length

def truncate_to_min_length(X, min_length):
    truncated_X = [arr[:, :min_length] if arr.shape[1] > min_length else arr for arr in X]
    print(f"truncate_to_min_length - min_length: {min_length}")
    return np.array(truncated_X)

def prepare_data(X, y, categories):
    try:
        # Ensure X contains NumPy arrays
        for col in X.columns:
            X[col] = X[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)
        
        # Verify if conversion was successful
        for col in X.columns:
            if not all(isinstance(val, np.ndarray) for val in X[col]):
                print(f"Column '{col}' contains non-NumPy array values.")
        
        # Find minimum length for truncation
        min_length = find_min_length(X.values.flatten())
        print(f"min_length: {min_length}")

        # Truncate data to minimum length
        X_truncated = truncate_to_min_length(X.values.flatten(), min_length)
        num_samples, num_features = X_truncated.shape
        print(f"X_truncated shape: {X_truncated.shape}")

        num_categories = len(categories)
        print(f"Number of categories: {num_categories}")

        y_encoded = pd.get_dummies(y).values
        print(f"y_encoded shape: {y_encoded.shape}")

        encoder = LabelEncoder()
        scaler = StandardScaler()
        
        return X_truncated, y_encoded, encoder, scaler

    except Exception as e:
        print(f"Error in prepare_data: {e}")
        raise




def build_and_train_model(X_train, y_train, X_test, y_test, num_features, num_classes):
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Print shapes and types for debugging
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_train dtype:", X_train.dtype)
    print("y_train dtype:", y_train.dtype)
    
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
        epochs=300, 
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

def main():
    full_xtract = 'df_output/v4_full_huge.csv'
    file_limit_2 = 'df_output/v4_file_depth_2.csv'
    
    def get_input_with_timeout(prompt, timeout=15):
        print(prompt, end='', flush=True)
        input_queue = queue.Queue()
        
        def input_thread():
            user_input = input()
            input_queue.put(user_input)
        
        thread = threading.Thread(target=input_thread)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            return 'N'
        else:
            return input_queue.get().strip().upper()
 
    try:
        df_extract = read_raw_str_csv_and_split_df(file_limit_2)

        if df_extract is not None:
            # Split into X and y
            X = df_extract.drop(columns=['filename', 'genre', 'harmony', 'perceptr', 'tempo'])
            y = df_extract['genre']
            categories = y.unique()
            num_classes = len(categories)

            print("Check X info: ")
            print(X.head())
            print("X columns: ")
            print(X.columns)
            print("Check y info: ")
            print(y.head())

            
            print(f"Type of X: {type(X)}")
            print(f"Type of y: {type(y)}")

            # Prepare the data
            X_scaled, y_encoded, encoder, scaler = prepare_data(X, y, categories)
            print("=======================> X AND Y SUCCESS")
            print(f"y_encoded shape: {y_encoded.shape}")
            print(f"Number of classes: {num_classes}")
            
            
            try:
                print(f"Type of X: {type(X)}")
                print(f"Type of y: {type(y)}")
                print(f"Type of X_scaled: {type(X_scaled)}")
                print(f"Type of y_encoded: {type(y_encoded)}")

                # Check specific variables
                print(f"X_scaled sample: {X_scaled[0]}")
                print(f"y_encoded sample: {y_encoded[0]}")
            except Exception as e:
                print(f"An error occurred while debugging types: {e}")     
                       

            y_encoded_one_hot = to_categorical(y_encoded, num_classes=num_classes)
            print(f"y_encoded_one_hot shape: {y_encoded_one_hot.shape}")

            if X_scaled is not None and y_encoded is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded_one_hot, test_size=0.2, random_state=42)
            else:
                print("Error in data preparation")
                raise ValueError("X_scaled or y_encoded is None")

            model, history = build_and_train_model(X_train, y_train, X_test, y_test, X_scaled.shape[1], num_classes)
            
            while True:
                try:
                    prompt_for_start_predictor = get_input_with_timeout("Would you like to predict a genre (Y/N)? ")
                    if prompt_for_start_predictor == 'Y':
                        row_input = get_input_with_timeout(f"Enter a row number (0 to {len(df_extract) - 1}), or Q to skip: ")
                        if row_input == 'Q':
                            break

                        try:
                            row_num = int(row_input)
                            if row_num < 0 or row_num >= len(df_extract):
                                raise ValueError("Row number is out of bounds.")
                        except ValueError as ve:
                            print(f"Error: {ve}")
                            continue

                        selected_row = X.iloc[row_num]  # Ensure X is still a DataFrame
                        selected_genre = y.iloc[row_num]  # Ensure y is still a Series
                        print(f"Selected row:\n{selected_row}")
                        print(f"Selected genre:\n{selected_genre}")

                        # Confirm to use this row for prediction
                        confirm = get_input_with_timeout("Do you want to use this song data for prediction? (Y/N), Enter for Y, or Q to quit: ")
                        if confirm in ('Y', ''):
                            # Make prediction
                            example_feature_inputs = selected_row.values  # Use .values for NumPy array
                            predicted_class = predict(model, encoder, scaler, example_feature_inputs)
                            print(f"Predicted class: {predicted_class}")
                        elif confirm == 'Q':
                            break
                        elif confirm != 'N':
                            print("Invalid input. Please enter Y or N, or use Q to quit.")
                    elif prompt_for_start_predictor == 'N':
                        break
                    else:
                        print("Invalid input. Please enter Y or N.")
                except Exception as e:
                    print(f"An error occurred in prediction block: {e}")
        else:
            print("Error: DataFrame is None")
   
        # Evaluate model
        evaluate_all_rows(model, X_scaled, y, encoder, scaler)
        
        # Save Pickles
        try:
            pickle_dir = 'pickles'
            if not os.path.exists(pickle_dir):
                os.makedirs(pickle_dir)
                print(f"Directory '{pickle_dir}' created.")
            else:
                print(f"Directory '{pickle_dir}' already exists.")
            
            model_path = os.path.join(pickle_dir, 'trained_model.pkl')
            
            with open(model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
            save_encoder_and_scaler(encoder, scaler)
            save_data(X_scaled, y)
            
            print(f"Model saved to '{model_path}'")
        except Exception as e:
            print(f"Error in Pickle: {e}")

    
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit()
        
        
    except Exception as e:
        print(f"An error occurred in main block: {e}")
    

if __name__ == '__main__':
    main()

