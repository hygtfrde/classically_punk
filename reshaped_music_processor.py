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


# columns_from_extracted_librosa_audio_data = [
#     'filename', 'genre', 'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz', 
#     'harmony', 'perceptr', 'tempo', 'zcr', 'rmse', 'spectral_bandwidth', 
#     'spectral_centroid', 'spectral_rolloff', 'spectral_flatness', 'onset', 
#     'beat', 'formants', 'pitch', 'loudness', 'autocorrelation'
# ]

# expected_types = {
#     'filename': 'object',
#     'genre': 'object',
#     'mfcc_mean': 'float32',
#     'mfcc_std': 'float32',
#     'chroma_mean': 'float32',
#     'chroma_std': 'float32',
#     'mel_mean': 'float32',
#     'mel_std': 'float32',
#     'contrast_mean': 'float32',
#     'contrast_std': 'float32',
#     'tonnetz_mean': 'float32',
#     'tonnetz_std': 'float32',
#     'harmony_mean': 'float32',
#     'harmony_std': 'float32',
#     'perceptr_mean': 'float32',
#     'perceptr_std': 'float32',
#     'tempo': 'float32',
#     'zcr_mean': 'float32',
#     'zcr_std': 'float32',
#     'rmse_mean': 'float32',
#     'rmse_std': 'float32',
#     'spectral_bandwidth_mean': 'float32',
#     'spectral_bandwidth_std': 'float32',
#     'spectral_centroid_mean': 'float32',
#     'spectral_centroid_std': 'float32',
#     'spectral_rolloff_mean': 'float32',
#     'spectral_rolloff_std': 'float32',
#     'spectral_flatness_mean': 'float32',
#     'spectral_flatness_std': 'float32',
#     'onset_mean': 'float32',
#     'onset_std': 'float32',
#     'beat_mean': 'float32',
#     'beat_std': 'float32',
#     'formants_mean': 'float32',
#     'formants_std': 'float32',
#     'pitch_mean': 'float32',
#     'pitch_std': 'float32',
#     'loudness_mean': 'float32',
#     'loudness_std': 'float32',
#     'autocorrelation_mean': 'float32',
#     'autocorrelation_std': 'float32'
# }

df_output_dir = 'df_output'

class MusicDataProcessor:
    def __init__(self, dataset_path: str, file_depth_limit: int, file_output_name: str):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.file_output_name = file_output_name
        self.genres = []  # Initialize genres list (populate it as needed)
        self.data = pd.DataFrame(columns=fundamental_features_cols)

        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")
            
    def get_data(self):
        self.data.to_csv(f'{df_output_dir}/{self.file_output_name}.csv', index=False)
        return self.data

    def extract_features(self, file_path, verbose='v', extract_raw_only=False):
        try:
            y, sr = librosa.load(file_path, sr=None)
            y = y[:min(int(30 * sr), len(y))]  # Extract first 30 seconds or less

            n_fft = min(1024, len(y))  # Ensure n_fft is not greater than the length of y

            def feature_stats(feature):
                mean = np.mean(feature, axis=1).astype(np.float32)
                std = np.std(feature, axis=1).astype(np.float32)
                return mean, std

            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

            if extract_raw_only:
                return {
                    'mfcc': mfcc.astype(np.float32).tolist(),
                    'chroma': chroma.astype(np.float32).tolist(),
                    'mel': mel.astype(np.float32).tolist(),
                    'contrast': contrast.astype(np.float32).tolist(),
                    'tonnetz': tonnetz.astype(np.float32).tolist()
                }

            # Extract Mean and StdDev for all features
            mfcc_mean, mfcc_std = feature_stats(mfcc)
            chroma_mean, chroma_std = feature_stats(chroma)
            mel_mean, mel_std = feature_stats(mel)
            contrast_mean, contrast_std = feature_stats(contrast)
            tonnetz_mean, tonnetz_std = feature_stats(tonnetz)

            if verbose == 'v':
                print(f"EXTRACTING: {file_path} \n"
                    f"  y: {y.shape}, dtype: {y.dtype} \n"
                    f"  sr: {sr} \n"
                    f"  mfcc_mean: {mfcc_mean.shape}, dtype: {mfcc_mean.dtype} \n"
                    f"  chroma_mean: {chroma_mean.shape}, dtype: {chroma_mean.dtype} \n"
                    f"  mel_mean: {mel_mean.shape}, dtype: {mel_mean.dtype} \n"
                    f"  contrast_mean: {contrast_mean.shape}, dtype: {contrast_mean.dtype} \n"
                    f"  tonnetz_mean: {tonnetz_mean.shape}, dtype: {tonnetz_mean.dtype} \n"
                    f"  mfcc_std: {mfcc_std.shape}, dtype: {mfcc_std.dtype} \n"
                    f"  chroma_std: {chroma_std.shape}, dtype: {chroma_std.dtype} \n"
                    f"  mel_std: {mel_std.shape}, dtype: {mel_std.dtype} \n"
                    f"  contrast_std: {contrast_std.shape}, dtype: {contrast_std.dtype} \n"
                    f"  tonnetz_std: {tonnetz_std.shape}, dtype: {tonnetz_std.dtype}")

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
                'tonnetz_std': tonnetz_std.tolist()
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
                if self.file_depth_limit and counter >= self.file_depth_limit:
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
                        'tonnetz_std': features['tonnetz_std']
                    })
                counter += 1

        self.data = pd.DataFrame(all_data)
        self.get_data()
    
    def debug_data(self):
        print("Original Data Types:")
        print(self.data.dtypes)
        print("DataFrame Info:")
        print(self.data.info())
        print("DataFrame Head:")
        print(self.data.head())
        print("DataFrame Tail:")
        print(self.data.tail())


# ------------------------------- MAIN -------------------------------
def main():
    dataset_path = 'genres'  # Replace with the path to your audio dataset
    file_depth_limit = 10  # Number of files to process per genre
    file_output_name = 'reshaped_music_data'  # Name for the output CSV file

    # Create an instance of the MusicDataProcessor
    processor = MusicDataProcessor(
        dataset_path=dataset_path,
        file_depth_limit=file_depth_limit,
        file_output_name=file_output_name
    )

    # Populate the genres list (assuming you have directories named by genre)
    processor.genres = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    # Load data
    processor.load_data()

    # Output the processed data
    print("Data has been processed and saved to CSV.")
    print(processor.data.head())  # Display the first few rows of the processed data

if __name__ == "__main__":
    main()