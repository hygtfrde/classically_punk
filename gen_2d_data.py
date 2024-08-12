import numpy as np
import pandas as pd

# Define the number of rows for the test dataset
num_rows = 30

# Define the size of the features
n_mfcc = 13
n_chroma = 12
n_mel = 128
n_contrast = 7
n_tonnetz = 6
n_bandwidth = 1
n_flatness = 1
n_centroid = 1
n_zero_crossing = 1
n_harmony = 1
n_perceptr = 1
n_rolloff = 1
n_rms = 1
n_frames = 100  # Number of frames for 2D features

def generate_random_feature_data():
    features = {
        'mfcc': np.random.rand(n_mfcc, n_frames).tolist(),  # 13 MFCCs, 100 frames
        'chroma': np.random.rand(n_chroma, n_frames).tolist(),  # 12 chroma bands, 100 frames
        'mel': np.random.rand(n_mel, n_frames).tolist(),  # 128 Mel bands, 100 frames
        'contrast': np.random.rand(n_contrast, n_frames).tolist(),  # 7 spectral contrast bands, 100 frames
        'tonnetz': np.random.rand(n_tonnetz, n_frames).tolist(),  # 6 tonnetz features, 100 frames
        'spectral_bandwidth': np.random.rand(n_bandwidth, n_frames).tolist(),  # 1 spectral bandwidth feature, 100 frames
        'spectral_flatness': np.random.rand(n_flatness, n_frames).tolist(),  # 1 spectral flatness feature, 100 frames
        'spectral_centroid': np.random.rand(n_centroid, n_frames).tolist(),  # 1 spectral centroid feature, 100 frames
        'zero_crossing_rate': np.random.rand(n_zero_crossing, n_frames).tolist(),  # 1 zero crossing rate feature, 100 frames
        'harmony': np.random.rand(n_harmony, n_frames).tolist(),  # 1 harmony feature, 100 frames
        'perceptr': np.random.rand(n_perceptr, n_frames).tolist(),  # 1 percussive feature, 100 frames
        'tempo': np.random.rand(n_perceptr).tolist(),  # 1 tempo value
        'spectral_rolloff': np.random.rand(n_rolloff, n_frames).tolist(),  # 1 spectral rolloff feature, 100 frames
        'rms': np.random.rand(n_rms, n_frames).tolist()  # 1 RMS feature, 100 frames
    }
    return features

# Create a DataFrame
data = []
for i in range(num_rows):
    features = generate_random_feature_data()
    row = {
        'filename': f'file_{i}.wav',
        'genre': np.random.choice(['Rock', 'Jazz', 'Pop', 'Classical', 'Hip-Hop']),
    }
    # Flatten the feature dict into the row dict
    for key, value in features.items():
        row[key] = str(value)  # Convert the 2D array to a string
    data.append(row)

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('test_2d_features.csv', index=False)
