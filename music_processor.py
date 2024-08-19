import os
import time
import json

import pandas as pd
import numpy as np
import librosa
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.stats import kurtosis, skew



# Constants
genres_from_dataset = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
fundamental_features_cols = [
    'mfcc', 'chroma', 'mel', 'contrast', 'tonnetz'
]

df_output_dir = 'df_output'

class MusicDataProcessor:
    def __init__(
            self, 
            dataset_path: str, 
            file_depth_limit: int, 
            file_output_name: str, 
            extract_raw_only: bool,
            compute_kde: bool,
            compute_ecdf: bool,
            pad_and_truncate: bool
        ):
        self.dataset_path = dataset_path
        self.file_depth_limit = file_depth_limit
        self.file_output_name = file_output_name
        self.genres = genres_from_dataset
        self.data = pd.DataFrame(columns=fundamental_features_cols)
        self.extract_raw_only = extract_raw_only
        self.compute_kde = compute_kde
        self.compute_ecdf = compute_ecdf
        self.pad_and_truncate = pad_and_truncate

        if not os.path.exists(df_output_dir):
            os.makedirs(df_output_dir)
            print(f"Directory '{df_output_dir}' created.")
        else:
            print(f"Directory '{df_output_dir}' already exists.")

    def get_data(self):
        def encode_array(x):
            if isinstance(x, np.ndarray):
                return json.dumps(x.tolist())  # Convert the array to a JSON string
            return x
        
        # NP saves
        # np.savez(f'{df_output_dir}/raw_features_{self.file_output_name}.npz', **features)
        
        # Apply the encoding to all elements in the DataFrame
        encoded_df = self.data.map(encode_array)
        # Save the DataFrame to CSV
        encoded_df.to_csv(f'{df_output_dir}/{self.file_output_name}.csv', index=False)
        
        return encoded_df

    def compute_stats_and_measures(self, data):
        # Compute basic statistics
        stats_dict = {
            'mean': np.mean(data),
            'stddev': np.std(data),
            'var': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'mad': stats.median_abs_deviation(data),
            'kurtosis': kurtosis(data),
            'skewness': skew(data)
        }
        
        # Compute ECDF
        if self.compute_ecdf:
            sorted_data, ecdf = np.sort(data), np.arange(1, len(data) + 1) / len(data)
            stats_dict['ecdf_values'] = sorted_data.tolist()
            stats_dict['ecdf_proportions'] = ecdf.tolist()
        
        # Compute KDE
        if self.compute_kde:
            kde = stats.gaussian_kde(data)
            stats_dict['kde'] = kde
        
        return stats_dict

    def extract_features(self, file_path, verbose='v'):
        try:
            target_rows = 13
            target_columns = 1293
            y, sr = librosa.load(file_path, sr=None)
            n_fft = min(1024, len(y))
            
            def pad_or_truncate(feature, target_columns):
                # Truncate
                if feature.shape[1] > target_columns:
                    return feature[:, :target_columns]
                # Pad
                elif feature.shape[1] < target_columns:
                    pad_width = target_columns - feature.shape[1]
                    return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
                return feature

            features = {
                'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=target_rows, n_fft=n_fft),
                'chroma': librosa.feature.chroma_stft(y=y, sr=sr, hop_length=n_fft // 4),
                'mel': librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft),
                'contrast': librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft),
                'tonnetz': librosa.feature.tonnetz(y=y, sr=sr),
                'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft),
                'spectral_flatness': librosa.feature.spectral_flatness(y=y),
                'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft),
                'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y),
                'harmony': librosa.effects.harmonic(y).reshape(1, -1),  # Reshape to 2D array
                'perceptr': librosa.effects.percussive(y).reshape(1, -1),  # Reshape to 2D array
                'tempo': np.array([librosa.beat.beat_track(y=y, sr=sr)[0]]).reshape(1, 1),  # Ensure shape compatibility
                'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft),
                'rms': librosa.feature.rms(y=y, frame_length=n_fft)
            }
            
            if self.pad_and_truncate:
                for key in features:
                    if len(features[key].shape) == 2:
                        features[key] = pad_or_truncate(features[key], target_columns)
                    else:
                        # Handle 1D features (e.g., tempo, harmony)
                        features[key] = pad_or_truncate(features[key].reshape(1, -1), target_columns)

            
            if self.extract_raw_only is not None and self.extract_raw_only:
                if verbose == 'v':
                    for name, array in features.items():
                        print(f"{name.capitalize()} Shape: {array.shape}")
                return features

            # Compute statistics for each feature
            feature_stats = {}
            for feature_name, feature_array in features.items():
                if feature_array.ndim == 1:  # If the feature is 1D
                    feature_stats.update({
                        f'{feature_name}_mean': np.mean(feature_array),
                        f'{feature_name}_stddev': np.std(feature_array),
                        f'{feature_name}_var': np.var(feature_array),
                        f'{feature_name}_min': np.min(feature_array),
                        f'{feature_name}_max': np.max(feature_array)
                    })
                else:  # If the feature is 2D
                    num_features = feature_array.shape[0]
                    for i in range(num_features):
                        feature_i = feature_array[i, :]
                        feature_stats.update({
                            f'{feature_name}_{i+1}_{key}': value
                            for key, value in self.compute_stats_and_measures(feature_i).items()
                        })

            if verbose == 'v':
                for key, value in feature_stats.items():
                    print(f"EXTRACTING: {key}\n{value}")

            return feature_stats

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None



    def load_data(self):
        all_data = []
        total_files_counter = 0
        for genre in self.genres:
            counter = 0
            genre_dir = os.path.join(self.dataset_path, genre)
            for file in os.listdir(genre_dir):
                print(f'File number: {total_files_counter}')
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
                    total_files_counter += 1

        self.data = pd.DataFrame(all_data)
        self.get_data()

# ------------------------------- MAIN -------------------------------
def main():
    # Start the timer
    start_time = time.time()
    
    dataset_path = 'genres'  # Replace with the path to your audio dataset
    file_depth_limit = None  # Number of files to process per genre
    file_output_name = 'v5_kde_full_all_stats'

    # Create an instance of the MusicDataProcessor
    processor = MusicDataProcessor(
        dataset_path=dataset_path,
        file_output_name=file_output_name, 
        file_depth_limit=file_depth_limit,
        extract_raw_only=False,
        pad_and_truncate=True,
        compute_kde=True,
        compute_ecdf=True
    )

    # Load data
    processor.load_data()

    # Output the processed data
    print(f"Data has been processed and saved to CSV file: {file_output_name}.")
    print(processor.data.head())  # Display the first few rows of the processed data
    
    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"Time taken: {minutes} minutes and {seconds} seconds")

if __name__ == "__main__":
    main()
