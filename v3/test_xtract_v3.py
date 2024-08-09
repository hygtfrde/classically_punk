import os
import numpy as np
import pandas as pd
import librosa

# Constants
genres_from_dataset = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
fundamental_features_cols = [
    'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz'
]
df_output_dir = 'df_output'

def extract_song_stats(dataset_path: str, genre: str, row_number: int, file_output_name: str):
    try:
        genre_dir = os.path.join(dataset_path, genre)
        files = os.listdir(genre_dir)
        if row_number >= len(files):
            print("Row number exceeds the number of files in the genre directory.")
            return None

        file_path = os.path.join(genre_dir, files[row_number])
        y, sr = librosa.load(file_path, sr=None)
        n_fft = min(1024, len(y))

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        mfcc_stats = {f'mfcc_{i+1}_mean': np.mean(mfcc[i, :]) for i in range(13)}
        chroma_stats = {f'chroma_{i+1}_mean': np.mean(chroma[i, :]) for i in range(chroma.shape[0])}
        mel_stats = {f'mel_{i+1}_mean': np.mean(mel[i, :]) for i in range(mel.shape[0])}
        contrast_stats = {f'contrast_{i+1}_mean': np.mean(contrast[i, :]) for i in range(contrast.shape[0])}
        tonnetz_stats = {f'tonnetz_{i+1}_mean': np.mean(tonnetz[i, :]) for i in range(tonnetz.shape[0])}

        stats = {**mfcc_stats, **chroma_stats, **mel_stats, **contrast_stats, **tonnetz_stats}
        stats['filename'] = files[row_number]
        stats['genre'] = genre

        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(f'{df_output_dir}/{file_output_name}_stats.csv', index=False)

        print(f"Extracted stats for {genre}/{files[row_number]} saved to {file_output_name}_stats.csv")

        return stats

    except Exception as e:
        print(f"Error in extract_song_stats: {e}")
        return None

def validate_stats(dataset_path: str, genre: str, row_number: int, file_output_name: str):
    try:
        genre_dir = os.path.join(dataset_path, genre)
        files = os.listdir(genre_dir)
        if row_number >= len(files):
            print("Row number exceeds the number of files in the genre directory.")
            return None

        file_path = os.path.join(genre_dir, files[row_number])
        y, sr = librosa.load(file_path, sr=None)
        n_fft = min(1024, len(y))

        # Extract MFCC for validation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)

        mfcc_means = {f'mfcc_{i+1}_mean': np.mean(mfcc[i, :]) for i in range(13)}

        validate_df = pd.DataFrame([mfcc_means])
        validate_df.to_csv(f'{df_output_dir}/{file_output_name}_validate.csv', index=False)

        print(f"Validation stats for {genre}/{files[row_number]} saved to {file_output_name}_validate.csv")

        return mfcc_means

    except Exception as e:
        print(f"Error in validate_stats: {e}")
        return None

# Example usage
dataset_path = 'genres'  # Replace with the path to your dataset
genre = 'blues'
row_number = 0  # First song in the genre
file_output_name = 'single_song_stats'

# Extract stats for a single song
extract_song_stats(dataset_path, genre, row_number, file_output_name)

# Validate stats for the same song
validate_stats(dataset_path, genre, row_number, file_output_name)
