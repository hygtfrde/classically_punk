import os
import ast

import pandas as pd
import numpy as np
import librosa


"""
    DATA COLUMNS SUMMARY
    - filename: The name of the audio file.
    - genre: The genre of the audio file.
    - mfcc: Mel-Frequency Cepstral Coefficients, which represent the short-term power spectrum of the audio file.
    - chroma: Chroma features, which relate to the twelve different pitch classes.
    - mel: Mel spectrogram, which represents the power of a signal in the mel scale frequencies.
    - contrast: Spectral contrast, which measures the difference in amplitude between peaks and valleys in a sound spectrum.
    - tonnetz: Tonnetz features, which capture harmonic and tonal properties.
    - harmony: Harmonic features of the audio.
    - perceptr: Perceptual features.
    - tempo: The tempo of the audio file.
"""



genres_from_dataset = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
columns_from_extracted_librosa_audio_data = [
    'filename', 'genre', 'mfcc_mean', 'mfcc_std', 'chroma_mean', 'chroma_std',
    'mel_mean', 'mel_std', 'contrast_mean', 'contrast_std', 'tonnetz_mean', 'tonnetz_std',
    'harmony_mean', 'harmony_std', 'perceptr_mean', 'perceptr_std', 'tempo'
]

expected_types = {
    'filename': 'object',
    'genre': 'object',
    'mfcc_mean': 'object',
    'mfcc_std': 'object',
    'chroma_mean': 'object',
    'chroma_std': 'object',
    'mel_mean': 'object',
    'mel_std': 'object',
    'contrast_mean': 'object',
    'contrast_std': 'object',
    'tonnetz_mean': 'object',
    'tonnetz_std': 'object',
    'harmony_mean': 'float32',
    'harmony_std': 'float32',
    'perceptr_mean': 'float32',
    'perceptr_std': 'float32',
    'tempo': 'object'
}

df_output_dir = 'df_output'

class MusicDataProcessor:
    def __init__(self, dataset_path: str, file_depth_limit: int, file_output_name: str):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.file_output_name = file_output_name
        self.genres = genres_from_dataset
        self.data = pd.DataFrame(columns=columns_from_extracted_librosa_audio_data)

        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")
            
    def get_data(self):
        self.data.to_csv(f'{df_output_dir}/test.csv', index=False)
        return self.data



    def extract_features(self, file_path, verbose='v', extract_raw_only=False):
        try:
            y, sr = librosa.load(file_path, sr=None)
            y = y[:min(int(30 * sr), len(y))]  # Extract first 30 seconds or less

            n_fft = min(1024, len(y))  # Ensure n_fft is not greater than the length of y

            def feature_stats(feature):
                mean = np.mean(feature).astype(np.float32)
                std = np.std(feature).astype(np.float32)
                return mean, std

            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            harmony = librosa.effects.harmonic(y)
            perceptr = librosa.effects.percussive(y)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            if extract_raw_only:
                return {
                    'mfcc': mfcc.astype(np.float32).tolist(),
                    'chroma': chroma.astype(np.float32).tolist(),
                    'mel': mel.astype(np.float32).tolist(),
                    'contrast': contrast.astype(np.float32).tolist(),
                    'tonnetz': tonnetz.astype(np.float32).tolist(),
                    'harmony': harmony.astype(np.float32).tolist(),
                    'perceptr': perceptr.astype(np.float32).tolist(),
                    'tempo': float(tempo)
                }

            # Extract Mean and StdDev for all features
            mfcc_mean, mfcc_std = feature_stats(mfcc)
            chroma_mean, chroma_std = feature_stats(chroma)
            mel_mean, mel_std = feature_stats(mel)
            contrast_mean, contrast_std = feature_stats(contrast)
            tonnetz_mean, tonnetz_std = feature_stats(tonnetz)
            harmony_mean, harmony_std = np.mean(harmony).astype(np.float32), np.std(harmony).astype(np.float32)
            perceptr_mean, perceptr_std = np.mean(perceptr).astype(np.float32), np.std(perceptr).astype(np.float32)

            if verbose == 'v':
                print(f"EXTRACTING: {file_path} \n"
                    f"  y: {y.shape}, dtype: {y.dtype} \n"
                    f"  sr: {sr} \n"
                    f"  mfcc_mean: {mfcc_mean.shape}, dtype: {mfcc_mean.dtype} \n"
                    f"  chroma_mean: {chroma_mean.shape}, dtype: {chroma_mean.dtype} \n"
                    f"  mel_mean: {mel_mean.shape}, dtype: {mel_mean.dtype} \n"
                    f"  contrast_mean: {contrast_mean.shape}, dtype: {contrast_mean.dtype} \n"
                    f"  tonnetz_mean: {tonnetz_mean.shape}, dtype: {tonnetz_mean.dtype} \n"
                    f"  harmony_mean: {harmony_mean:.10f}, dtype: {type(harmony_mean)} \n"
                    f"  harmony_std: {harmony_std:.10f}, dtype: {type(harmony_std)} \n"
                    f"  perceptr_mean: {perceptr_mean:.10f}, dtype: {type(perceptr_mean)} \n"
                    f"  perceptr_std: {perceptr_std:.10f}, dtype: {type(perceptr_std)} \n"
                    f"  tempo: {tempo}, dtype: {type(tempo)}")

            return {
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist(),
                'chroma_mean': chroma_mean.tolist(),
                'chroma_std': chroma_std.tolist(),
                'mel_mean': mel_mean.tolist(),
                'mel_std': mel_std.tolist(),
                'contrast_mean': contrast_mean.tolist(),
                'contrast_std': contrast_std.tolist(),
                'tonnetz_mean': tonnetz_mean.tolist(),
                'tonnetz_std': tonnetz_std.tolist(),
                'harmony_mean': harmony_mean,
                'harmony_std': harmony_std,
                'perceptr_mean': perceptr_mean,
                'perceptr_std': perceptr_std,
                'tempo': float(tempo)
            }

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None


    def load_data(self):
        all_data = []
        for genre in self.genres:
            counter = 0
            genre_dir = os.path.join(self.dataset_path, genre)
            for file in os.listdir(genre_dir):
                if counter >= self.file_depth_limit:
                    break
                file_path = os.path.join(genre_dir, file)
                
                features = self.extract_features(file_path, 'v')
                if features is not None:
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        'mfcc_mean': features['mfcc_mean'],
                        'mfcc_std': features['mfcc_std'],
                        'chroma_mean': features['chroma_mean'],
                        'chroma_std': features['chroma_std'],
                        'mel_mean': features['mel_mean'],
                        'mel_std': features['mel_std'],
                        'contrast_mean': features['contrast_mean'],
                        'contrast_std': features['contrast_std'],
                        'tonnetz_mean': features['tonnetz_mean'],
                        'tonnetz_std': features['tonnetz_std'],
                        'harmony_mean': features['harmony_mean'],
                        'harmony_std': features['harmony_std'],
                        'perceptr_mean': features['perceptr_mean'],
                        'perceptr_std': features['perceptr_std'],
                        'tempo': features['tempo']
                    })
                counter += 1

        self.data = pd.DataFrame(all_data)
        self.get_data()
        
        # VALIDATE AFTER LOADING
        # validate_data_after_loading = self.validate_data()
        # print('Is data validated after loading? => ', validate_data_after_loading)


    def validate_data(self):
        valid = True
        print('VALIDATING DATA EXTRACTION')
        print("DataFrame Info:")
        print(self.data.info())
        print("\nDataFrame Head:")
        print(self.data.head())
        print("\nDataFrame Tail:")
        print(self.data.tail())
        
        # Check for missing values
        if self.data.isnull().values.any():
            valid = False
            print("Error: Data contains missing values.")
            print(self.data[self.data.isnull().any(axis=1)])  # Display rows with missing values

        # Validate genre column
        if 'genre' in self.data.columns:
            if not all(self.data['genre'].isin(self.genres)):
                valid = False
                print("Error: Data contains invalid genres.")
                invalid_genres = self.data[~self.data['genre'].isin(self.genres)]['genre'].unique()
                print(f"Invalid genres: {invalid_genres}")
        else:
            valid = False
            print("Error: 'genre' column is missing from the DataFrame.")

        # Validate column data types
        for column, expected_type in expected_types.items():
            if column in self.data.columns:
                if self.data[column].dtype != expected_type:
                    valid = False
                    print(f"Validation Error: Column '{column}' has type '{self.data[column].dtype}' but expected '{expected_type}'.")
            else:
                valid = False
                print(f"Validation Error: Column '{column}' is missing from the DataFrame.")

        # Load from CSV
        if os.path.isfile(f'{df_output_dir}/test.csv'):
            df_loaded = pd.read_csv(f'{df_output_dir}/test.csv')
        else:
            df_loaded = None

        # Check if DataFrames are equal
        print('CHECKING EQUAL self.data vs df_loaded ===========================> ', self.data.equals(df_loaded))
        # Print info for further inspection
        print('👇 CURRENT EXTRACTED DF 👇')
        print(self.data.info())
        print('👇 LOADED DF (CSV, XLSX, PARQUET) 👇')
        print(df_loaded.info())
        print('===========================')

        return valid

    
    def debug_data(self):
        print("Original Data Types:")
        print(self.data.dtypes)
        print("DataFrame Info:")
        print(self.data.info())
        print("DataFrame Head:")
        print(self.data.head())
        print("DataFrame Tail:")
        print(self.data.tail())
