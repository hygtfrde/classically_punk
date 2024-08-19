import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import librosa
import librosa.display

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
    def plot_mfccs(self, mfcc_mean, mfcc_std, filename):
        # Plotting mean and standard deviation
        plt.figure(figsize=(12, 6))
        plt.errorbar(np.arange(len(mfcc_mean)), mfcc_mean, yerr=mfcc_std, fmt='-o', capsize=5)
        plt.title(f'MFCC Mean and Std Dev - {filename}')
        plt.xlabel('MFCC Coefficients')
        plt.ylabel('Mean Value')
        plt.savefig(f'visualizations/extracted_audio_data/{filename}_mfccs.png')
        plt.close()
        
    def plot_chroma(self, chroma_mean, chroma_std, filename):
        # Plotting mean and standard deviation
        plt.figure(figsize=(12, 6))
        plt.errorbar(np.arange(len(chroma_mean)), chroma_mean, yerr=chroma_std, fmt='-o', capsize=5)
        plt.title(f'Chroma Mean and Std Dev - {filename}')
        plt.xlabel('Chroma Coefficients')
        plt.ylabel('Mean Value')
        plt.savefig(f'visualizations/extracted_audio_data/{filename}_chroma.png')
        plt.close()

    def plot_feature(self, feature_mean, feature_std, feature_name, filename):
        plt.figure(figsize=(12, 6))
        plt.errorbar(np.arange(len(feature_mean)), feature_mean, yerr=feature_std, fmt='-o', capsize=5)
        plt.title(f'{feature_name} Mean and Std Dev - {filename}')
        plt.xlabel(f'{feature_name} Coefficients')
        plt.ylabel('Mean Value')
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
        visualized_counts = {}
        for idx, row in self.df.iterrows():
            genre = row['genre']
            if visualized_counts.get(genre, 0) >= rows_visualized_per_genre:
                continue
            filename = row['filename']

            # Extract features from the DataFrame
            mfcc_mean = row['mfcc_mean']
            mfcc_std = row['mfcc_std']
            chroma_mean = row['chroma_mean']
            chroma_std = row['chroma_std']
            mel_mean = row['mel_mean']
            mel_std = row['mel_std']
            contrast_mean = row['contrast_mean']
            contrast_std = row['contrast_std']
            tonnetz_mean = row['tonnetz_mean']
            tonnetz_std = row['tonnetz_std']
            harmony_mean = row['harmony_mean']
            harmony_std = row['harmony_std']
            perceptr_mean = row['perceptr_mean']
            perceptr_std = row['perceptr_std']
            tempo = row['tempo']

            # Convert features to numpy arrays if necessary for plotting
            mfcc_mean = np.array(mfcc_mean)
            mfcc_std = np.array(mfcc_std)
            chroma_mean = np.array(chroma_mean)
            chroma_std = np.array(chroma_std)
            mel_mean = np.array(mel_mean)
            mel_std = np.array(mel_std)
            contrast_mean = np.array(contrast_mean)
            contrast_std = np.array(contrast_std)
            tonnetz_mean = np.array(tonnetz_mean)
            tonnetz_std = np.array(tonnetz_std)

            # Plot and save visualizations
            self.plot_mfccs(mfcc_mean, mfcc_std, filename)
            self.plot_chroma(chroma_mean, chroma_std, filename)
            self.plot_feature(mel_mean, mel_std, 'Mel', filename)
            self.plot_feature(contrast_mean, contrast_std, 'Contrast', filename)
            self.plot_feature(tonnetz_mean, tonnetz_std, 'Tonnetz', filename)
            
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
