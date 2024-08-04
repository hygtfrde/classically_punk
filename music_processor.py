import os
import pandas as pd
import numpy as np
import librosa


genres_from_dataset = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
columns_from_extracted_librosa_audio_data = ['filename', 'genre', 'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz', 'harmony_mean', 'harmony_std', 'perceptr_mean', 'perceptr_std', 'tempo']

expected_types = {
    'filename': 'object',
    'genre': 'object',
    'mfcc': 'object',  # Stored as lists or arrays
    'chroma': 'object',  # Stored as lists or arrays
    'mel': 'object',  # Stored as lists or arrays
    'contrast': 'object',  # Stored as lists or arrays
    'tonnetz': 'object',  # Stored as lists or arrays
    'harmony_mean': 'float32',
    'harmony_std': 'float32',
    'perceptr_mean': 'float32',
    'perceptr_std': 'float32',
    'tempo': 'object'  # Assuming it will be a single float value
}


class MusicDataProcessor:
    def __init__(self, dataset_path: str, file_depth_limit: int, file_output_name: str):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.file_output_name = file_output_name
        self.genres = genres_from_dataset
        self.data = pd.DataFrame(columns=columns_from_extracted_librosa_audio_data)

    def extract_features(self, file_path, verbose=1):
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            harmony = librosa.effects.harmonic(y)
            perceptr = librosa.effects.percussive(y)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Extract Mean and StdDev from harmony and perceptr
            harmony_mean = np.mean(harmony).astype(np.float32)
            harmony_std = np.std(harmony).astype(np.float32)
            perceptr_mean = np.mean(perceptr).astype(np.float32)
            perceptr_std = np.std(perceptr).astype(np.float32)

            # Debug output with increased precision
            # np.set_printoptions(precision=10, suppress=True)  # Set precision and suppress scientific notation
            print(f"Harmony: Min={np.min(harmony):.10f}, Max={np.max(harmony):.10f}, Range={np.ptp(harmony):.10f}")
            print(f"Perciptr: Min={np.min(perceptr):.10f}, Max={np.max(perceptr):.10f}, Range={np.ptp(perceptr):.10f}")

            if verbose == 1:
                print(f"EXTRACTING: {file_path} \n"
                      f"  y: {y.shape}, dtype: {y.dtype} \n"
                      f"  sr: {sr} \n"
                      f"  mfcc: {mfcc.shape}, dtype: {mfcc.dtype} \n"
                      f"  chroma: {chroma.shape}, dtype: {chroma.dtype} \n"
                      f"  mel: {mel.shape}, dtype: {mel.dtype} \n"
                      f"  contrast: {contrast.shape}, dtype: {contrast.dtype} \n"
                      f"  tonnetz: {tonnetz.shape}, dtype: {tonnetz.dtype} \n"
                      f"  harmony_mean: {harmony_mean:.10f}, dtype: {type(harmony_mean)} \n"
                      f"  harmony_std: {harmony_std:.10f}, dtype: {type(harmony_std)} \n"
                      f"  perceptr_mean: {perceptr_mean:.10f}, dtype: {type(perceptr_mean)} \n"
                      f"  perceptr_std: {perceptr_std:.10f}, dtype: {type(perceptr_std)} \n"
                      f"  tempo: {tempo}, dtype: {type(tempo)}")
            return mfcc, chroma, mel, contrast, tonnetz, harmony_mean, harmony_std, perceptr_mean, perceptr_std, tempo

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
                features = self.extract_features(file_path, 1)
                if features is not None:
                    mfcc, chroma, mel, contrast, tonnetz, harmony_mean, harmony_std, perceptr_mean, perceptr_std, tempo = features
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        'mfcc': mfcc.tolist(),  # Convert arrays to lists if needed
                        'chroma': chroma.tolist(),
                        'mel': mel.tolist(),
                        'contrast': contrast.tolist(),
                        'tonnetz': tonnetz.tolist(),
                        'harmony_mean': harmony_mean,  # Store mean and std as separate values
                        'harmony_std': harmony_std,
                        'perceptr_mean': perceptr_mean,
                        'perceptr_std': perceptr_std,
                        'tempo': tempo
                    })
                counter += 1

        self.data = pd.DataFrame(all_data)
        
        validate_data_after_loading = self.validate_data()
        print('Is data validated after loading? => ', validate_data_after_loading)

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
        if not all(self.data['genre'].isin(self.genres)):
            valid = False
            print("Error: Data contains invalid genres.")
            invalid_genres = self.data[~self.data['genre'].isin(self.genres)]['genre'].unique()
            print(f"Invalid genres: {invalid_genres}")
        
        for column, expected_type in expected_types.items():
            if self.data[column].dtype != expected_type:
                valid = False
                print(f"Validation Error: Column '{column}' has type '{self.data[column].dtype}' but expected '{expected_type}'.")

        return valid
    
    def coerce_data_types(self, df):
        for column, expected_type in expected_types.items():
            if column in df.columns:
                try:
                    if expected_type == 'object' and column == 'tempo':
                        df[column] = df[column].apply(lambda x: np.array(x) if isinstance(x, str) else x)
                    else:
                        df[column] = df[column].astype(expected_type)
                except ValueError as e:
                    print(f"Error converting column {column} to {expected_type}: {e}")
        return df

    def get_data(self):
        df_output_dir = 'df_output'
        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")
        
                
        # self.data.to_excel(file_path, index=False)
        # self.data.to_csv(f'{df_output_dir}/_default_data.csv', index=False)
        # self.data.to_parquet(f'{df_output_dir}/_default_data.parquet', engine='fastparquet')
        
        # Save to CSV
        # self.data = self.coerce_data_types(self.data)
        self.data.to_csv(f'{df_output_dir}/test.csv', index=False)
        # Load from CSV
        df_loaded = pd.read_csv(f'{df_output_dir}/test.csv')
        # df_loaded = self.coerce_data_types(df_loaded)

        # Check if DataFrames are equal
        print('CHECKING EQUAL self.data vs df_loaded ===========================> ', self.data.equals(df_loaded))
        # Print info for further inspection
        print('👇 CURRENT EXTRACTED DF 👇')
        print(self.data.info())
        print('👇 LOADED DF (CSV, XLSX, PARQUET) 👇')
        print(df_loaded.info())
        print('===========================')


        return self.data
    
    def debug_data(self):
        print("Original Data Types:")
        print(self.data.dtypes)
        print("DataFrame Info:")
        print(self.data.info())
        print("DataFrame Head:")
        print(self.data.head())
        print("DataFrame Tail:")
        print(self.data.tail())
