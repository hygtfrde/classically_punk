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
    def __init__(self, dataset_path: str, file_depth_limit: int, file_output_name: str, extract_raw_only: bool, verbose: str):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.file_output_name = file_output_name
        self.genres = genres_from_dataset
        self.data = pd.DataFrame(columns=fundamental_features_cols)
        self.extract_raw_only = extract_raw_only
        self.verbose = verbose

        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")


    def get_data(self):
        self.data.to_csv(f'{df_output_dir}/{self.file_output_name}.csv', index=False)
        return self.data
    

    def extract_features(self, file_path):
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
            
            if self.extract_raw_only is not None:
                return {
                    'mfcc': mfcc,
                    'chroma': chroma,
                    'mel': mel,
                    'contrast': contrast,
                    'tonnetz': tonnetz
                }

            def compute_stats(features, name_prefix):
                stats = {}
                for i in range(features.shape[0]):
                    feature_i = features[i, :]
                    stats[f'{name_prefix}_{i+1}_mean'] = np.mean(feature_i)
                    stats[f'{name_prefix}_{i+1}_stddev'] = np.std(feature_i)
                    stats[f'{name_prefix}_{i+1}_var'] = np.var(feature_i)
                    stats[f'{name_prefix}_{i+1}_min'] = np.min(feature_i)
                    stats[f'{name_prefix}_{i+1}_max'] = np.max(feature_i)
                return stats

            # Compute stats for each feature set
            mfcc_stats = compute_stats(mfcc, 'mfcc')
            chroma_stats = compute_stats(chroma, 'chroma')
            mel_stats = compute_stats(mel, 'mel')
            contrast_stats = compute_stats(contrast, 'contrast')
            tonnetz_stats = compute_stats(tonnetz, 'tonnetz')

            if self.verbose == 'v' or 'V':
                print(f"EXTRACTING: mfcc_stats\n{mfcc_stats}")
                print(f"EXTRACTING: chroma_stats\n{chroma_stats}")
                print(f"EXTRACTING: mel_stats\n{mel_stats}")
                print(f"EXTRACTING: contrast_stats\n{contrast_stats}")
                print(f"EXTRACTING: tonnetz_stats\n{tonnetz_stats}")

            all_stats = {**mfcc_stats, **chroma_stats, **mel_stats, **contrast_stats, **tonnetz_stats}
            return all_stats

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
    file_depth_limit = None  # Number of files to process per genre
    file_output_name = 'all_songs_5_stat_sets'  # Name for the output CSV file

    # Create an instance of the MusicDataProcessor
    processor = MusicDataProcessor(
        dataset_path=dataset_path,
        file_output_name=file_output_name, 
        file_depth_limit=file_depth_limit,
        extract_raw_only=True
    )

    # Load data
    processor.load_data()

    # Output the processed data
    print(f"Data has been processed and saved to CSV file: {file_output_name}.")
    print(processor.data.head())  # Display the first few rows of the processed data

if __name__ == "__main__":
    main()