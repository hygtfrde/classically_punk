import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def visualize_audio_features(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(['filename', 'genre'], axis=1)
    
    for col in df.columns:
        data = df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
        
        plt.figure(figsize=(10, 4))
        
        # For columns with 2D features
        if col in ['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz']:
            data_2d = np.stack(data.values)
            if data_2d.ndim == 3 and data_2d.shape[2] == 1293:  # Ensure it's a 2D array for each time step
                librosa.display.specshow(data_2d.mean(axis=0), x_axis='time', cmap='coolwarm')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{col.upper()} visualization')
        
        # For columns with 1D features
        elif col in ['spectral_bandwidth', 'spectral_flatness', 'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff', 'rms', 'tempo']:
            data_1d = np.concatenate(data.values)
            if data_1d.ndim == 1:
                plt.plot(data_1d, color='r')
                plt.title(f'{col.replace("_", " ").capitalize()} visualization')
                plt.xlabel('Time')
                plt.ylabel(col.replace("_", " ").capitalize())
        
        # For columns like harmony and perceptr, check dimensions
        elif col in ['harmony', 'perceptr']:
            data_generic = np.stack(data.values)
            if data_generic.ndim == 3 and data_generic.shape[2] == 1293:  # 2D for each time step
                librosa.display.specshow(data_generic.mean(axis=0), x_axis='time', cmap='coolwarm')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{col.capitalize()} visualization')
            elif data_generic.ndim == 2:  # 1D array
                plt.plot(data_generic.flatten(), color='g')
                plt.title(f'{col.capitalize()} visualization')
                plt.xlabel('Time')
                plt.ylabel(col.capitalize())
        
        # Save plot to the plots directory
        plot_path = os.path.join('plots', f'{col}_visualization.png')
        plt.savefig(plot_path)
        plt.close()

if not os.path.exists('plots'):
    os.makedirs('plots')

csv_file = 'df_output/v5_5.csv'
visualize_audio_features(csv_file)
