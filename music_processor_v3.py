import os

import pandas as pd
import numpy as np
import librosa
from sklearn.decomposition import PCA
import scipy.stats as stats



# Constants
genres_from_dataset = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
fundamental_features_cols = [
    'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz'
]

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
            n_fft = min(1024, len(y))

            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
            
            if self.extract_raw_only is not None and self.extract_raw_only:
                # Save raw features to CSV for full inspection
                np.savez(f'{df_output_dir}/raw_features_{self.file_output_name}.npz', mfcc=mfcc, chroma=chroma, mel=mel, contrast=contrast, tonnetz=tonnetz)
                
                if verbose == 'v':
                    print(f"MFCC Shape: {mfcc.shape}")
                    print(f"Chroma Shape: {chroma.shape}")
                    print(f"Mel Shape: {mel.shape}")
                    print(f"Contrast Shape: {contrast.shape}")
                    print(f"Tonnetz Shape: {tonnetz.shape}")
                
                return {
                    'mfcc': mfcc,
                    'chroma': chroma,
                    'mel': mel,
                    'contrast': contrast,
                    'tonnetz': tonnetz
                }

            mfcc_stats = {}
            for i in range(13):
                mfcc_i = mfcc[i, :]
                mfcc_stats[f'mfcc_{i+1}_mean'] = np.mean(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_stddev'] = np.std(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_var'] = np.var(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_min'] = np.min(mfcc_i)
                mfcc_stats[f'mfcc_{i+1}_max'] = np.max(mfcc_i)
            
            chroma_stats = {}
            num_chroma_features = chroma.shape[0]
            num_chroma_frames = chroma.shape[1]
            for i in range(min(13, num_chroma_features)):  # Avoid out-of-bounds error
                chroma_i = chroma[i, :]
                chroma_stats[f'chroma_{i+1}_mean'] = np.mean(chroma_i)
                chroma_stats[f'chroma_{i+1}_stddev'] = np.std(chroma_i)
                chroma_stats[f'chroma_{i+1}_var'] = np.var(chroma_i)
                chroma_stats[f'chroma_{i+1}_min'] = np.min(chroma_i)
                chroma_stats[f'chroma_{i+1}_max'] = np.max(chroma_i)

            mel_stats = {}
            num_mel_features = mel.shape[0]
            for i in range(num_mel_features):
                mel_i = mel[i, :]
                mel_stats[f'mel_{i+1}_mean'] = np.mean(mel_i)
                mel_stats[f'mel_{i+1}_stddev'] = np.std(mel_i)
                mel_stats[f'mel_{i+1}_var'] = np.var(mel_i)
                mel_stats[f'mel_{i+1}_min'] = np.min(mel_i)
                mel_stats[f'mel_{i+1}_max'] = np.max(mel_i)

            contrast_stats = {}
            num_contrast_features = contrast.shape[0]
            for i in range(num_contrast_features):
                contrast_i = contrast[i, :]
                contrast_stats[f'contrast_{i+1}_mean'] = np.mean(contrast_i)
                contrast_stats[f'contrast_{i+1}_stddev'] = np.std(contrast_i)
                contrast_stats[f'contrast_{i+1}_var'] = np.var(contrast_i)
                contrast_stats[f'contrast_{i+1}_min'] = np.min(contrast_i)
                contrast_stats[f'contrast_{i+1}_max'] = np.max(contrast_i)

            tonnetz_stats = {}
            num_tonnetz_features = tonnetz.shape[0]
            for i in range(num_tonnetz_features):
                tonnetz_i = tonnetz[i, :]
                tonnetz_stats[f'tonnetz_{i+1}_mean'] = np.mean(tonnetz_i)
                tonnetz_stats[f'tonnetz_{i+1}_stddev'] = np.std(tonnetz_i)
                tonnetz_stats[f'tonnetz_{i+1}_var'] = np.var(tonnetz_i)
                tonnetz_stats[f'tonnetz_{i+1}_min'] = np.min(tonnetz_i)
                tonnetz_stats[f'tonnetz_{i+1}_max'] = np.max(tonnetz_i)
                
            spectral_bandwidth_stats = {}
            num_bandwidth_features = spectral_bandwidth.shape[0]
            for i in range(num_bandwidth_features):
                bandwidth_i = spectral_bandwidth[i, :]
                spectral_bandwidth_stats[f'spectral_bandwidth_{i+1}_mean'] = np.mean(bandwidth_i)
                spectral_bandwidth_stats[f'spectral_bandwidth_{i+1}_stddev'] = np.std(bandwidth_i)
                spectral_bandwidth_stats[f'spectral_bandwidth_{i+1}_var'] = np.var(bandwidth_i)
                spectral_bandwidth_stats[f'spectral_bandwidth_{i+1}_min'] = np.min(bandwidth_i)
                spectral_bandwidth_stats[f'spectral_bandwidth_{i+1}_max'] = np.max(bandwidth_i)
                
            spectral_flatness_stats = {}
            num_flatness_features = spectral_flatness.shape[0]
            for i in range(num_flatness_features):
                flatness_i = spectral_flatness[i, :]
                spectral_flatness_stats[f'spectral_flatness_{i+1}_mean'] = np.mean(flatness_i)
                spectral_flatness_stats[f'spectral_flatness_{i+1}_stddev'] = np.std(flatness_i)
                spectral_flatness_stats[f'spectral_flatness_{i+1}_var'] = np.var(flatness_i)
                spectral_flatness_stats[f'spectral_flatness_{i+1}_min'] = np.min(flatness_i)
                spectral_flatness_stats[f'spectral_flatness_{i+1}_max'] = np.max(flatness_i)

            spectral_centroid_stats = {}
            num_centroid_features = spectral_centroid.shape[0]
            for i in range(num_centroid_features):
                centroid_i = spectral_centroid[i, :]
                spectral_centroid_stats[f'spectral_centroid_{i+1}_mean'] = np.mean(centroid_i)
                spectral_centroid_stats[f'spectral_centroid_{i+1}_stddev'] = np.std(centroid_i)
                spectral_centroid_stats[f'spectral_centroid_{i+1}_var'] = np.var(centroid_i)
                spectral_centroid_stats[f'spectral_centroid_{i+1}_min'] = np.min(centroid_i)
                spectral_centroid_stats[f'spectral_centroid_{i+1}_max'] = np.max(centroid_i)

            zero_crossing_rate_stats = {}
            num_zero_crossing_features = zero_crossing_rate.shape[0]
            for i in range(num_zero_crossing_features):
                zero_crossing_i = zero_crossing_rate[i, :]
                zero_crossing_rate_stats[f'zero_crossing_rate_{i+1}_mean'] = np.mean(zero_crossing_i)
                zero_crossing_rate_stats[f'zero_crossing_rate_{i+1}_stddev'] = np.std(zero_crossing_i)
                zero_crossing_rate_stats[f'zero_crossing_rate_{i+1}_var'] = np.var(zero_crossing_i)
                zero_crossing_rate_stats[f'zero_crossing_rate_{i+1}_min'] = np.min(zero_crossing_i)
                zero_crossing_rate_stats[f'zero_crossing_rate_{i+1}_max'] = np.max(zero_crossing_i)


            if verbose == 'v':
                print(f"EXTRACTING: mfcc_stats\n{mfcc_stats}")
                print(f"EXTRACTING: chroma_stats\n{chroma_stats}")
                print(f"EXTRACTING: mel_stats\n{mel_stats}")
                print(f"EXTRACTING: contrast_stats\n{contrast_stats}")
                print(f"EXTRACTING: tonnetz_stats\n{tonnetz_stats}")
                print(f"EXTRACTING: spectral_bandwidth_stats\n{spectral_bandwidth_stats}")
                print(f"EXTRACTING: spectral_flatness_stats\n{spectral_flatness_stats}")
                print(f"EXTRACTING: spectral_centroid_stats\n{spectral_centroid_stats}")
                print(f"EXTRACTING: zero_crossing_rate_stats\n{zero_crossing_rate_stats}")


            return {
                **mfcc_stats, 
                **chroma_stats, 
                **mel_stats, 
                **contrast_stats, 
                **tonnetz_stats, 
                **spectral_bandwidth_stats, 
                **spectral_flatness_stats,
                **spectral_centroid_stats,
                **zero_crossing_rate_stats
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
                print(f'File number: {counter}')
                if self.file_depth_limit and counter >= self.file_depth_limit:
                    break
                file_path = os.path.join(genre_dir, file)
                features = self.extract_features(file_path, 'v')
                if features:
                    # Flatten and unpack the data structure
                    stats_flat = features
                    all_data.append({
                        'filename': file,
                        'genre': genre,
                        **stats_flat
                    })                                      
                    counter += 1

        self.data = pd.DataFrame(all_data)
        self.get_data()

# ------------------------------- MAIN -------------------------------
def main():
    dataset_path = 'genres'  # Replace with the path to your audio dataset
    file_depth_limit = None  # Number of files to process per genre
    file_output_name = 'v3_full_xtract_all_songs'  # Name for the output CSV file

    # Create an instance of the MusicDataProcessor
    processor = MusicDataProcessor(
        dataset_path=dataset_path,
        file_output_name=file_output_name, 
        file_depth_limit=file_depth_limit,
        extract_raw_only=None
    )

    # Load data
    processor.load_data()

    # Output the processed data
    print(f"Data has been processed and saved to CSV file: {file_output_name}.")
    print(processor.data.head())  # Display the first few rows of the processed data

if __name__ == "__main__":
    main()
