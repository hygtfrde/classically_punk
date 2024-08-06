import pandas as pd
import numpy as np
import json

# Define the genres
genres = [
    'blues', 'classical', 'country', 'disco', 
    'hiphop', 'jazz', 'metal', 'pop', 
    'reggae', 'rock'
]

# Number of features
num_mfcc = 13
num_chroma = 12
num_mel = 128
num_contrast = 7
num_tonnetz = 6

# Function to generate random feature arrays
def generate_random_features(num):
    return json.dumps(np.random.rand(num).tolist())

# Generate dummy data
data = []
for genre in genres:
    row = {
        'filename': f'{genre}.00001.wav',
        'genre': genre,
        'mfcc_mean': generate_random_features(num_mfcc),
        'mfcc_std': generate_random_features(num_mfcc),
        'chroma_mean': generate_random_features(num_chroma),
        'chroma_std': generate_random_features(num_chroma),
        'mel_mean': generate_random_features(num_mel),
        'mel_std': generate_random_features(num_mel),
        'contrast_mean': generate_random_features(num_contrast),
        'contrast_std': generate_random_features(num_contrast),
        'tonnetz_mean': generate_random_features(num_tonnetz),
        'tonnetz_std': generate_random_features(num_tonnetz),
        'harmony_mean': generate_random_features(1),
        'harmony_std': generate_random_features(1),
        'perceptr_mean': generate_random_features(1),
        'perceptr_std': generate_random_features(1),
        'tempo': np.random.rand()
    }
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('dummy_music.csv', index=False)

# Print the DataFrame
print(df)