import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import librosa


class AudioDataVisualizer:
    def __init__(self, df):
        self.df = df
        self.ensure_directories()

    def ensure_directories(self):
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        if not os.path.exists('visualizations/_test_audio_waveform_spectogram'):
            os.makedirs('visualizations/_test_audio_waveform_spectogram')
        if not os.path.exists('visualizations/extracted_audio_data'):
            os.makedirs('visualizations/extracted_audio_data')

    # ------------------------------- AUDIO FILE VISUALIZERS
    def plot_waveform(self, audio_data, sample_rate, filename):
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
        plt.title(f'Waveform - {filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig(f'visualizations/_test_audio_waveform_spectogram/{filename}_waveform.png')
        plt.close()

    def plot_spectrogram(self, audio_data, sample_rate, filename):
        frequencies, times, Sxx = spectrogram(audio_data, sample_rate)
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
        plt.title(f'Spectrogram - {filename}')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Intensity [dB]')
        plt.savefig(f'visualizations/_test_audio_waveform_spectogram/{filename}_spectrogram.png')
        plt.close()
    # -------------------------------

    # ------------------------------- LIBROSA AUDIO DATA VISUALIZERS
    def plot_mfccs(self, mfccs, filename):
        # Ensure mfccs is a numerical NumPy array
        mfccs = np.array(mfccs, dtype=np.float32)  # Convert to float32 for numerical operations

        # Handle potential non-finite values
        mfccs = np.nan_to_num(mfccs, copy=False)  # Replace NaN and inf with 0

        # Ensure all values in mfccs are finite
        mfccs = np.where(np.isfinite(mfccs), mfccs, 0)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCC - {filename}')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.savefig(f'visualizations/extracted_audio_data/{filename}_mfccs.png')
        plt.close()
        
    def plot_chroma(self, chroma, filename):
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(chroma, x_axis='time')
        plt.colorbar()
        plt.title(f'Chroma - {filename}')
        plt.xlabel('Time')
        plt.ylabel('Chroma Coefficients')
        plt.savefig(f'visualizations/extracted_audio_data/{filename}_chroma.png')
        plt.close()
        
    def plot_feature(self, feature, feature_name, filename):
        plt.figure(figsize=(12, 6))

        if feature_name in ['Chroma', 'Mel', 'Contrast', 'Tonnetz']:
            # Assuming these features are 2D
            if feature.ndim == 2:
                librosa.display.specshow(feature, x_axis='time')
                plt.colorbar()
            else:
                raise ValueError(f'Feature for {feature_name} must be a 2D array.')
        elif feature_name in ['Harmony', 'Perceptr']:
            # Assuming these features are 1D
            print(f'{feature_name} => feature.ndim: {feature.ndim}')
            if feature.ndim == 1:
                self.plot_1d_feature(feature, feature_name, filename)
            else:
                raise ValueError(f'Feature for {feature_name} must be a 1D array.')
        else:
            raise ValueError(f'Unsupported feature type: {feature_name}')
        
        plt.savefig(f'visualizations/extracted_audio_data/{filename}_{feature_name.lower()}.png')
        plt.close()

    def plot_1d_feature(self, feature, feature_name, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(feature)
        plt.title(f'{feature_name} - {filename}')
        plt.xlabel('Time')
        plt.ylabel(f'{feature_name} Values')
        plt.savefig(f'visualizations/extracted_audio_data/{filename}_{feature_name.lower()}.png')
        plt.close()

    def plot_scalar_feature(self, value, feature_name, filename):
        plt.figure(figsize=(8, 4))
        plt.bar(feature_name, value)
        plt.title(f'{feature_name} - {filename}')
        plt.ylabel(f'{feature_name} Value')
        plt.savefig(f'visualizations/extracted_audio_data/{filename}_{feature_name.lower()}.png')
        plt.close()
    # -------------------------------

    def visualize(self, rows_visualized_per_genre):
        # Plot Test Audio File Waveform and Spectogram
        test_audio_file_1 = 'genres/blues/blues.00000.wav'
        audio_data, sample_rate = librosa.load(test_audio_file_1, sr=None)
        self.plot_waveform(audio_data, sample_rate, 'test_blues_00000')
        self.plot_spectrogram(audio_data, sample_rate, 'test_blues_00000')
        
        # Plots for each genre limited by rows_visualized_per_genre
        visualized_counts = {}
        for idx, row in self.df.iterrows():
            genre = row['genre']
            if visualized_counts.get(genre, 0) >= rows_visualized_per_genre:
                continue
            filename = row['filename']

            # Extract features from the DataFrame
            mfccs = row['mfcc']
            chroma = row['chroma']
            mel = row['mel']
            contrast = row['contrast']
            tonnetz = row['tonnetz']
            harmony_mean = row['harmony_mean']
            harmony_std = row['harmony_std']
            perceptr_mean = row['perceptr_mean']
            perceptr_std = row['perceptr_std']
            tempo = row['tempo']

            # Convert features to numpy arrays if necessary for plotting
            mfccs = np.array(mfccs)
            chroma = np.array(chroma)
            mel = np.array(mel)
            contrast = np.array(contrast)
            tonnetz = np.array(tonnetz)

            # Plot and save visualizations
            self.plot_mfccs(mfccs, filename)
            self.plot_feature(chroma, 'Chroma', filename)
            self.plot_feature(mel, 'Mel', filename)
            self.plot_feature(contrast, 'Contrast', filename)
            self.plot_feature(tonnetz, 'Tonnetz', filename)
            
            # Plot scalar features
            self.plot_scalar_feature(harmony_mean, 'Harmony Mean', filename)
            self.plot_scalar_feature(harmony_std, 'Harmony Std', filename)
            self.plot_scalar_feature(perceptr_mean, 'Perceptr Mean', filename)
            self.plot_scalar_feature(perceptr_std, 'Perceptr Std', filename)
            self.plot_scalar_feature(tempo, 'Tempo', filename)

            visualized_counts[genre] = visualized_counts.get(genre, 0) + 1
            
            print(f'Visualizations for {filename} saved.')

            # Stop if the limit is reached for this genre
            if visualized_counts[genre] >= rows_visualized_per_genre:
                print(f'Reached limit of {rows_visualized_per_genre} visualizations for genre: {genre}')
                continue
