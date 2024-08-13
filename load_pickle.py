import os
import pickle
from main_model_v4 import evaluate_all_rows, predict  


def load_model(pickle_file_path):
    try:
        with open(pickle_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"Model loaded from '{pickle_file_path}'")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_encoder_and_scaler(encoder_path='pickles/encoder.pkl', scaler_path='pickles/scaler.pkl'):
    try:
        with open(encoder_path, 'rb') as enc_file:
            encoder = pickle.load(enc_file)
        with open(scaler_path, 'rb') as scal_file:
            scaler = pickle.load(scal_file)
        print("Encoder and scaler loaded successfully.")
        return encoder, scaler
    except Exception as e:
        print(f"Error loading encoder and scaler: {e}")
        return None, None

def load_data(X_scaled_path='pickles/X_scaled.pkl', y_path='pickles/y.pkl'):
    try:
        with open(X_scaled_path, 'rb') as x_file:
            X_scaled = pickle.load(x_file)
        with open(y_path, 'rb') as y_file:
            y = pickle.load(y_file)
        print("Data loaded successfully.")
        return X_scaled, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def prompt_for_feature(name):
    while True:
        try:
            value = float(input(f"Enter value for {name}: "))
            return value
        except ValueError:
            print(f"Invalid input for {name}. Please enter a numeric value.")

def main():
    pickle_dir = 'pickles'
    model_path = os.path.join(pickle_dir, 'trained_model.pkl')
    encoder_path = os.path.join(pickle_dir, 'encoder.pkl')
    scaler_path = os.path.join(pickle_dir, 'scaler.pkl')
    X_scaled_path = os.path.join(pickle_dir, 'X_scaled.pkl')
    y_path = os.path.join(pickle_dir, 'y.pkl')

    model = load_model(model_path)
    encoder, scaler = load_encoder_and_scaler(encoder_path, scaler_path)
    X_scaled, y = load_data(X_scaled_path, y_path)

    if model and encoder and scaler and X_scaled is not None and y is not None:
        mfcc = prompt_for_feature('mfcc')
        chroma = prompt_for_feature('chroma')
        mel = prompt_for_feature('mel')
        contrast = prompt_for_feature('contrast')
        tonnetz = prompt_for_feature('tonnetz')
        spectral_bandwidth = prompt_for_feature('spectral_bandwidth')
        spectral_flatness = prompt_for_feature('spectral_flatness')
        spectral_centroid = prompt_for_feature('spectral_centroid')
        zero_crossing_rate = prompt_for_feature('zero_crossing_rate')
        harmony = prompt_for_feature('harmony')
        perceptr = prompt_for_feature('perceptr')
        tempo = prompt_for_feature('tempo')
        spectral_rolloff = prompt_for_feature('spectral_rolloff')
        rms = prompt_for_feature('rms')

        example_feature_inputs = [
            mfcc, chroma, mel, contrast, tonnetz, spectral_bandwidth,
            spectral_flatness, spectral_centroid, zero_crossing_rate,
            harmony, perceptr, tempo, spectral_rolloff, rms
        ]

        predicted_class = predict(model, encoder, scaler, example_feature_inputs)
        print(f"Predicted class: {predicted_class}")

        evaluate_all_rows(model, X_scaled, y, encoder, scaler)

if __name__ == '__main__':
    main()

