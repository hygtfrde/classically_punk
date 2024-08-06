import pandas as pd
import numpy as np

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

# Function to generate a single random integer for each feature
def generate_random_feature():
    return np.random.randint(1, 1001)

# Function to generate dummy data with specified rows per genre
def generate_dummy_data(num_rows_per_genre=1):
    data = []
    for genre in genres:
        for _ in range(num_rows_per_genre):
            row = {
                'filename': f'{genre}.{np.random.randint(1, 10000):05d}.wav',
                'genre': genre,
                'mfcc_mean': generate_random_feature(),
                'mfcc_std': generate_random_feature(),
                'chroma_mean': generate_random_feature(),
                'chroma_std': generate_random_feature(),
                'mel_mean': generate_random_feature(),
                'mel_std': generate_random_feature(),
                'contrast_mean': generate_random_feature(),
                'contrast_std': generate_random_feature(),
                'tonnetz_mean': generate_random_feature(),
                'tonnetz_std': generate_random_feature(),
                'harmony_mean': generate_random_feature(),
                'harmony_std': generate_random_feature(),
                'perceptr_mean': generate_random_feature(),
                'perceptr_std': generate_random_feature(),
                'tempo': generate_random_feature()
            }
            data.append(row)
    return pd.DataFrame(data)

# Generate and save dummy data
df = generate_dummy_data(num_rows_per_genre=100)  # Change the number of rows per genre here
df.to_csv('dummy_music.csv', index=False)

# Print the DataFrame
print(df)
