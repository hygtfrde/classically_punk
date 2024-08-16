import ast
import sys
import os
import pickle

# TENSORFLOW IS REQUIRED EVEN IF NOT ACCESSED
import tensorflow as tf

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from scikeras.wrappers import KerasClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


from CONSTANTS import RED, GREEN, RESET


# ---------------------------------------------------------

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
            
            if col not in ['filename', 'genre']:
                df_input[col] = df_input[col].apply(convert_string_to_array)
        
        # print('BEGIN SHAPE TEST ----------------------------------------------------- ')
        # X = df_input.drop(columns=['filename', 'genre', 'harmony', 'perceptr', 'tempo'])
        # for col in X.columns:
        #     print(f"Column '{col}', dtype: {X[col].dtype}")
        #     for value in X[col].head(5):
        #         print(f"Value type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")
        # print('END SHAPE TEST ----------------------------------------------------- ')

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



def create_keras_model(num_features, num_classes):
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model
    
    
# ---------------------------------------------------------



def bain():
    smaller_test = 'df_output/v5_3_file.csv'
    
    try:
        df_extract = read_raw_str_csv_and_split_df(smaller_test)
        
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
        
            print('======================= CHECK SHAPES =======================')
            print(f'X_train shape: {X_train.shape}')
            print(f'y_train shape: {y_train.shape}')
            print(f'X_test shape: {X_test.shape}')
            print(f'y_test shape: {y_test.shape}')
            print('======================= CHECK SHAPES =======================')
        
            # Build model using KerasClassifier
            model = KerasClassifier(model=create_keras_model, num_features=X_scaled.shape[1], num_classes=num_classes, epochs=300, batch_size=128, verbose=1)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Convert y_test and y_pred to class labels
            y_test_labels = np.argmax(y_test, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)

            # Accuracy
            accuracy = accuracy_score(y_test_labels, y_pred_labels)
            print(f"Accuracy: {accuracy:.4f}")
            
            print("Classification Report:")
            print(classification_report(y_test_labels, y_pred_labels, zero_division=1))
            
            print("Confusion Matrix:")
            print(confusion_matrix(y_test_labels, y_pred_labels))

            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"Cross-Validation Accuracy Scores: {scores}")
            print(f"Mean Accuracy: {scores.mean():.4f}")

            # Test accuracy
            test_accuracy = model.score(X_test, y_test)
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # Train pipeline
            pipeline = make_pipeline(StandardScaler(), LogisticRegression())
            pipeline.fit(X_train, y_encoded)
            
            # Evaluate on test data
            pipeline_score = pipeline.score(X_test, y_encoded[np.isin(y_encoded, y_test_labels)])
            print(f"Pipeline Test Accuracy: {pipeline_score:.4f}")

            # Grid search
            param_grid = {'C': [0.1, 1, 10]}
            grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_encoded)
            
            # Best parameters and score
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

        else:
            print("Error: DataFrame is None")
        
    except Exception as e:
        print(f"A general error occurred in main block: {e}")
        
        
if __name__ == '__main__':
    bain()