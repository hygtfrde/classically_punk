import os

import librosa
import numpy as np
import pandas as pd


df_output_dir = 'df_output'

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

        # Extract features for validation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)


        # Compute stats for other features
        def compute_stats(feature_array, feature_name):
            stats = {}
            num_features = feature_array.shape[0]
            for i in range(min(13, num_features)):  # To match MFCC's 13 coefficients
                feature_i = feature_array[i, :]
                stats.update({
                    f'{feature_name}_{i+1}_mean': np.mean(feature_i),
                    f'{feature_name}_{i+1}_stddev': np.std(feature_i),
                    f'{feature_name}_{i+1}_var': np.var(feature_i),
                    f'{feature_name}_{i+1}_min': np.min(feature_i),
                    f'{feature_name}_{i+1}_max': np.max(feature_i)
                })
            return stats

        mfcc_stats = compute_stats(mfcc, 'mfcc')
        chroma_stats = compute_stats(chroma, 'chroma')
        mel_stats = compute_stats(mel, 'mel')
        contrast_stats = compute_stats(contrast, 'contrast')
        tonnetz_stats = compute_stats(tonnetz, 'tonnetz')

        # Combine all stats into one dictionary
        combined_stats = {
            'filename': files[row_number],
            'genre': genre,
            **mfcc_stats,
            **chroma_stats,
            **mel_stats,
            **contrast_stats,
            **tonnetz_stats
        }

        # Convert to DataFrame and save to CSV
        validate_df = pd.DataFrame([combined_stats])
        validate_df.to_csv(f'{df_output_dir}/{file_output_name}_validate.csv', index=False)

        print(f"Validation stats for {genre}/{files[row_number]} saved to {file_output_name}_validate.csv")

        return combined_stats

    except Exception as e:
        print(f"Error in validate_stats: {e}")
        return None
    
def main():
    validate_stats('genres', 'blues', 0, 'validate_blues_0')
    

if __name__ == '__main__':
    main()