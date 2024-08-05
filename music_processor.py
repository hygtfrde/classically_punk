import os
import pandas as pd
import numpy as np
import librosa

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

    def extract_features(self, file_path, verbose='v'):
        try:
            y, sr = librosa.load(file_path, sr=None)
            start_time = 0
            end_time = 30  # Extract first 30 seconds

            # Convert start and end times to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Ensure end_sample is within the length of y
            end_sample = min(end_sample, len(y))

            # Slice audio snippet
            y = y[start_sample:end_sample]
            
            # Handle cases where y is too short
            n_fft = 1024
            if len(y) < n_fft:
                n_fft = len(y)  # Adjust n_fft to the length of y

            # Extract features from the snippet
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            harmony = librosa.effects.harmonic(y)
            perceptr = librosa.effects.percussive(y)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Extract Mean and StdDev for all features
            mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
            mfcc_std = np.std(mfcc, axis=1).astype(np.float32)
            chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
            chroma_std = np.std(chroma, axis=1).astype(np.float32)
            mel_mean = np.mean(mel, axis=1).astype(np.float32)
            mel_std = np.std(mel, axis=1).astype(np.float32)
            contrast_mean = np.mean(contrast, axis=1).astype(np.float32)
            contrast_std = np.std(contrast, axis=1).astype(np.float32)
            tonnetz_mean = np.mean(tonnetz, axis=1).astype(np.float32)
            tonnetz_std = np.std(tonnetz, axis=1).astype(np.float32)
            harmony_mean = np.mean(harmony).astype(np.float32)
            harmony_std = np.std(harmony).astype(np.float32)
            perceptr_mean = np.mean(perceptr).astype(np.float32)
            perceptr_std = np.std(perceptr).astype(np.float32)

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

            return mfcc_mean, mfcc_std, chroma_mean, chroma_std, mel_mean, mel_std, contrast_mean, contrast_std, tonnetz_mean, tonnetz_std, harmony_mean, harmony_std, perceptr_mean, perceptr_std, tempo

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
                
                # Define snippet start and end times (in seconds)
                start_time = 0  # Start at beginning
                end_time = 30   # Extract first 30 seconds of each file
                
                features = self.extract_features(file_path, 'v')
                if features is not None:
                    mfcc_mean, mfcc_std, chroma_mean, chroma_std, mel_mean, mel_std, contrast_mean, contrast_std, tonnetz_mean, tonnetz_std, harmony_mean, harmony_std, perceptr_mean, perceptr_std, tempo = features
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        'mfcc_mean': mfcc_mean.tolist(),  # Convert arrays to lists if needed
                        'mfcc_std': mfcc_std.tolist(),
                        'chroma_mean': chroma_mean.tolist(),
                        'chroma_std': chroma_std.tolist(),
                        'mel_mean': mel_mean.tolist(),
                        'mel_std': mel_std.tolist(),
                        'contrast_mean': contrast_mean.tolist(),
                        'contrast_std': contrast_std.tolist(),
                        'tonnetz_mean': tonnetz_mean.tolist(),
                        'tonnetz_std': tonnetz_std.tolist(),
                        'harmony_mean': harmony_mean,  # Store mean and std as separate values
                        'harmony_std': harmony_std,
                        'perceptr_mean': perceptr_mean,
                        'perceptr_std': perceptr_std,
                        'tempo': tempo
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
