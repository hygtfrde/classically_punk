import ast
import threading
import queue
import sys
import json

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


# -----------------------------------------------------------------------------------------
def convert_string_to_array_v0(string):
    try:
        list_of_floats = ast.literal_eval(string)
        return np.array(list_of_floats, dtype=float)  # Ensure float type
    except Exception as e:
        print(f"Error converting string to array: {e}")
        return np.array([])


def read_csv_and_split_df(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['filename', 'genre'])  # Drop non-feature columns
    y = df['genre']  # Target column
    return X, y
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def read_large_csv_in_chunks(csv_path):
    chunk_size = 100
    chunks = []
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            # Apply conversion on each chunk
            for col in chunk.columns:
                if col not in ['filename', 'genre']:
                    chunk[col] = chunk[col].apply(convert_string_to_array)
            chunks.append(chunk)
        
        # Concatenate chunks
        df = pd.concat(chunks, ignore_index=True)
        # X = df.drop(columns=['filename', 'genre'])
        # y = df['genre']
        return df
    
    except Exception as e:
        print(f"Error processing CSV in chunks: {e}")
        return None, None
# -----------------------------------------------------------------------------------------






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
                    
                    # Flatten if it's a 2D array with one row
                    # if value.ndim == 2 and value.shape[0] == 1:
                    #     value = value.flatten()
                    
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
        
        X = df_input.drop(columns=['filename', 'genre', 'harmony', 'perceptr', 'tempo'])
        # y = df_input['genre']

        # Check shapes and types of the X values
        print('BEGIN SHAPE TEST ----------------------------------------------------- ')
        for col in X.columns:
            print(f"Column '{col}', dtype: {X[col].dtype}")
            for value in X[col].head(5):
                print(f"Value type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")
        print('END SHAPE TEST ----------------------------------------------------- ')

        return df_input
    else:
        print('Error: df_input is None')
        return None, None
    
    

    
    




# -----------------------------------------------------------------------------------------

# def prepare_data(X, y, categories):
#     encoder = OneHotEncoder(sparse_output=False, categories=[categories])
#     y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  # Convert to DataFrame with feature names
#     return X_scaled_df, y_encoded, encoder, scaler

# def prepare_data(X, y, categories):
#     try:
#         # Flatten the data if it contains nested arrays
#         X_flattened = pd.DataFrame()
#         for col in X.columns:
#             col_data = X[col]
#             # Handle 2D arrays by flattening or stacking
#             if isinstance(col_data.iloc[0], np.ndarray) and col_data.iloc[0].ndim > 1:
#                 col_data = col_data.apply(lambda x: x.flatten())
#             # Add flattened column to DataFrame
#             X_flattened[col] = col_data
        
#         # Scaling features
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(np.stack(X_flattened.to_numpy(), axis=1))
        
#         # Encoding labels
#         encoder = LabelEncoder()
#         y_encoded = encoder.fit_transform(y)
        
#         return X_scaled, y_encoded, encoder, scaler
#     except Exception as e:
#         print(f"Error in prepare_data: {e}")
#         return None, None, None, None

def prepare_data(X, y, categories):
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
    # Convert feature_inputs to DataFrame
    feature_df = pd.DataFrame([feature_inputs], columns=scaler.feature_names_in_)
    
    # Preprocess the feature inputs
    features_scaled = scaler.transform(feature_df)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    # Decode the predictions to category names
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = encoder.categories_[0][predicted_class_index]
    
    return predicted_class



def evaluate_all_rows(model, X, y, encoder, scaler):
    correct_count = 0
    total_count = len(X)
    
    for i in range(total_count):
        # Extract feature inputs and true label
        feature_inputs = X.iloc[i].values
        true_label = y.iloc[i]
        
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

    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.exit()
    # return accuracy, correct_count, incorrect_count
# -----------------------------------------------------------------------------------------





def main():
    full_xtract = 'df_output/v4_encoded_strings.csv'
    
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
            # Timeout occurred, we assume the user did not respond
            return 'N'
        else:
            return input_queue.get().strip().upper()
    
    try:  
        df_extract = read_raw_str_csv_and_split_df(full_xtract)
        
        # X = df_extract.drop(columns=['filename', 'genre', 'harmony', 'perceptr', 'tempo'])
        # y = df_extract['genre']
        # categories = y.unique()
        # num_classes = len(categories)
        # X_scaled, y_encoded, encoder, scaler = prepare_data(X, y, categories)
        # print("=======================> X AND Y SUCCESS")
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        
        
        
        if df_extract is not None:
            # Split into X and y
            X = df_extract.drop(columns=['filename', 'genre', 'harmony', 'perceptr', 'tempo'])
            y = df_extract['genre']
            categories = y.unique()
            num_classes = len(categories)

            print("Check X info:")
            print(X.head())
            print("Check y info:")
            print(y.head())

            # Prepare the data
            X_scaled, y_encoded, encoder, scaler = prepare_data(X, y, categories)
            print("=======================> X AND Y SUCCESS")
            print(f"y_encoded shape: {y_encoded.shape}")
            print(f"Number of classes: {num_classes}")
            
            y_encoded_one_hot = to_categorical(y_encoded, num_classes=num_classes)
            print(f"y_encoded_one_hot shape: {y_encoded_one_hot.shape}")

            if X_scaled is not None and y_encoded is not None:
                # Proceed with train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded_one_hot, test_size=0.2, random_state=42)
            else:
                print("Error in data preparation")
        else:
            print("Error: DataFrame is None")

        
        
        
        # if df_extract is not None:
        #     # Split into X and y
        #     X = df_extract.drop(columns=['filename', 'genre', 'harmony', 'perceptr', 'tempo'])
        #     y = df_extract['genre']
        #     categories = y.unique()
            
        #     print("Check X info:")
        #     print(X.head())
        #     print("Check y info:")
        #     print(y.head())
            
        #     print("Debug")
        #     print("Preview X before prepare_data:")
        #     for col in X.columns:
        #         print(f"Column: {col}, Type: {type(X[col].iloc[0])}, Shape: {getattr(X[col].iloc[0], 'shape', 'N/A')}")

        #     # Prepare the data
        #     X_scaled, y_encoded, encoder, scaler = prepare_data(X, y, categories)
            
            
        #     print("=======================> X AND Y SUCCESS")
            
        #     if X_scaled is not None and y_encoded is not None:
        #         # Proceed with train-test split
        #         X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        #     else:
        #         print("Error in data preparation")
        # else:
        #     print("Error: DataFrame is None")
        
        # Print data shapes
        # print(f"Training feature matrix shape: {X_train.shape}")
        # print(f"Testing feature matrix shape: {X_test.shape}")
        # print(f"Training target shape: {y_train.shape}")
        # print(f"Testing target shape: {y_test.shape}")
        
        
        # Build and train model
        model, history = build_and_train_model(X_train, y_train, X_test, y_test, X_scaled.shape[1], num_classes)
        
        
        # Print training history
        # print("Training history:")
        # for key in history.history.keys():
        #     print(f"{key}: {history.history[key]}")
        

        # Make Predictions with user input for row or song to test
        while True:
            try:
                prompt_for_start_predictor = get_input_with_timeout("Would you like to predict a genre (Y/N)? ")
                if prompt_for_start_predictor == 'Y':
                    row_input = get_input_with_timeout(f"Enter a row number (0 to {len(X) - 1}), or Q to skip: ")
                    if row_input == 'Q':
                        break  # Exit the loop if the user wants to skip

                    try:
                        row_num = int(row_input)
                        if row_num < 0 or row_num >= len(X):
                            raise ValueError("Row number is out of bounds.")
                    except ValueError as ve:
                        print(f"Error: {ve}")
                        continue  # Prompt user again if row number input is invalid

                    # Print the selected row
                    selected_row = X.iloc[row_num]
                    selected_genre = y.iloc[row_num]
                    print(f"Selected row:\n{selected_row}")
                    print(f"Selected genre:\n{GREEN}{selected_genre}{RESET}")

                    # Confirm to use this row for prediction
                    confirm = get_input_with_timeout("Do you want to use this song data for prediction? (Y/N), Enter for Y, or Q to quit: ")
                    if confirm in ('Y', ''):
                        # Make prediction
                        example_feature_inputs = selected_row.values
                        predicted_class = predict(model, encoder, scaler, example_feature_inputs)
                        print(f"Predicted class: {predicted_class}")
                    elif confirm == 'Q':
                        break  # Exit the loop if the user wants to quit
                    elif confirm != 'N':
                        print("Invalid input. Please enter Y or N, or use Q to quit.")
                elif prompt_for_start_predictor == 'N':
                    break  # Exit the loop if the user does not want to predict a genre
                else:
                    print("Invalid input. Please enter Y or N.")
            except Exception as e:
                print(f"An error occurred in prediction block: {e}")

        
        # Evaluate model
        evaluate_all_rows(model, X_scaled, y, encoder, scaler)
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit()
        
    except Exception as e:
        print(f"An error occurred in main block: {e}")




if __name__ == '__main__':
    main()
