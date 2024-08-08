import os
import ast

import pandas as pd
import numpy as np
import librosa
from sklearn.decomposition import PCA


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

fundamental_features_cols = [
    'mfcc',         # Mel-Frequency Cepstral Coefficients
    'chroma',       # Chroma Features
    'mel',          # Mel Spectrogram
    'contrast',     # Spectral Contrast
    'tonnetz'       # Tonnetz Features
]

expected_types_for_fundamental_features = {
    'mfcc': 'float32',        # 2D numpy array (coefficients x frames)
    'chroma': 'float32',      # 2D numpy array (pitch classes x frames)
    'mel': 'float32',         # 2D numpy array (mel bands x frames)
    'contrast': 'float32',    # 2D numpy array (bands x frames)
    'tonnetz': 'float32'      # 2D numpy array (tonnetz features x frames)
}

df_output_dir = 'df_output'



class MusicDataProcessor:
    def __init__(self, dataset_path: str, file_depth_limit: int, file_output_name: str, extract_raw_only: bool):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.file_output_name = file_output_name
        self.genres = genres_from_dataset
        self.data = pd.DataFrame(columns=fundamental_features_cols)
        self.extract_raw_only = extract_raw_only

        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")


    def get_data(self):
        self.data.to_csv(f'{df_output_dir}/{self.file_output_name}.csv', index=False)
        return self.data
    

    def extract_features(self, file_path, verbose='v'):
        try:
            y, sr = librosa.load(file_path, sr=None)

            # Set n_fft appropriately
            n_fft = min(1024, len(y))  # Ensure n_fft is not greater than the length of y

            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

            mfcc_stats = {}
            for i in range(13):
                mfcc_i = mfcc[i, :]
                mfcc_stats[f'mfcc_{i+1}_mean'] = np.mean(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_stddev'] = np.std(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_var'] = np.var(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_min'] = np.min(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_max'] = np.max(mfcc_i)

            return mfcc_stats

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None



    def load_data(self):
        all_data = []
        for genre in self.genres:
            counter = 0
            genre_dir = os.path.join(self.dataset_path, genre)
            for file in os.listdir(genre_dir):
                if self.file_depth_limit and counter >= self.file_depth_limit:
                    break
                file_path = os.path.join(genre_dir, file)
                features = self.extract_features(file_path, 'v')
                if features:
                    # Flatten and prepare the data structure
                    mfcc_stats_flat = features  # Directly use the mfcc_stats dictionary
                    
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        **mfcc_stats_flat
                    })
                    counter += 1

        self.data = pd.DataFrame(all_data)
        self.get_data()



# ------------------------------- MAIN -------------------------------
def main():
    dataset_path = 'genres'  # Replace with the path to your audio dataset
    file_depth_limit = 1  # Number of files to process per genre
    file_output_name = 'only_mfcc_1'  # Name for the output CSV file

    # Create an instance of the MusicDataProcessor
    processor = MusicDataProcessor(
        dataset_path=dataset_path,
        file_depth_limit=file_depth_limit,
        file_output_name=file_output_name, 
        extract_raw_only=True
    )

    # Load data
    processor.load_data()

    # Output the processed data
    print("Data has been processed and saved to CSV.")
    print(processor.data.head())  # Display the first few rows of the processed data

if __name__ == "__main__":
    main()