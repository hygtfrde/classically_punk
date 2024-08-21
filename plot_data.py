import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def visualize_audio_features(csv_file):
    df = pd.read_csv(csv_file)
    df = df[['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz']]
    
    for col in df.columns:
        data = df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
        
        plt.figure(figsize=(10, 4))
        
        data_2d = np.stack(data.values)
        if data_2d.ndim == 3 and data_2d.shape[2] == 1293:
            librosa.display.specshow(data_2d.mean(axis=0), x_axis='time', cmap='coolwarm')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'{col.upper()} visualization')
        
        # Save plot to the plots directory
        plot_path = os.path.join('plots', f'{col}_visualization.png')
        plt.savefig(plot_path)
        plt.close()

if not os.path.exists('plots'):
    os.makedirs('plots')

csv_file = 'df_output/v5_5.csv'
visualize_audio_features(csv_file)
