import ast
import threading
import queue
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

GREEN = '\033[32m'
RED = '\033[31m'
RESET = '\033[0m'



def convert_string_to_array(string):
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


def prepare_data(X, y, categories):
    encoder = OneHotEncoder(sparse_output=False, categories=[categories])
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  # Convert to DataFrame with feature names
    return X_scaled_df, y_encoded, encoder, scaler

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
    





def main():
    full_xtract = 'df_output/v3_full_no_kde.csv'
    test_csv_path = 'df_output/test_1.csv'
    
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
        # Read and prepare data
        X, y = read_csv_and_split_df(full_xtract)
        categories = y.unique()
        num_classes = len(categories)
        X_scaled, y_encoded, encoder, scaler = prepare_data(X, y, categories)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        
        # Print data shapes
        print(f"Training feature matrix shape: {X_train.shape}")
        print(f"Testing feature matrix shape: {X_test.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Testing target shape: {y_test.shape}")
        
        # Build and train model
        model, history = build_and_train_model(X_train, y_train, X_test, y_test, X_scaled.shape[1], num_classes)
        
        # Print training history
        print("Training history:")
        for key in history.history.keys():
            print(f"{key}: {history.history[key]}")
        

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
