import os

import pandas as pd
import librosa

from audio_vizualizer import AudioDataVisualizer
from music_processor import MusicDataProcessor

from helpers import get_user_input



# ------------------------------- GLOBAL VARS
test_audio_file_1 = 'genres/blues/blues.00000.wav'
dataset_path = 'genres'
default_csv_file_path = 'df_output/test_1.csv'

BLUE = '\033[34m'
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'





# ------------------------------- MAIN -------------------------------
def main():
    music_data = None
    print("Configure the following options:")
    process_data = get_user_input("Do you want to process the data? (Y/N): ", default_value=True)
    visualize_data = get_user_input("Do you want to visualize data? (Y/N): ", default_value=True)
    
    
    # ------------------------------- MusicDataProcessor
    if process_data:
        print(f"{BLUE}Begin Data Processing{RESET}")
        genre_classifier = MusicDataProcessor(dataset_path, None, 'test_1')

        print("Loading data...")
        genre_classifier.load_data()
        print(f"{GREEN}Data loaded successfully!{RESET}")

        print("Getting data...")
        music_data = genre_classifier.get_data()
        print('Music Data for Processor: \n', music_data)
    else: print('Skipping Data Processing')
    
    
    # ------------------------------- AudioDataVisualizer
    if visualize_data:
        if music_data is None:
            print('No currently processed data. Using Default DF for music data')
            if not os.path.exists(default_csv_file_path):
                print(f"Default DF file {default_csv_file_path} does not exist. Aborting.")
                return
            else:
                music_data = pd.read_csv(default_csv_file_path, engine='python')
                print(f'Music Data for Visualizer: \n', music_data)
        
        visualizer = AudioDataVisualizer(music_data)
        print("Plotting data...")
        # Plot Test Audio File Waveform and Spectogram of Test Audio
        audio_data, sample_rate = librosa.load(test_audio_file_1, sr=None)
        visualizer.plot_waveform(audio_data, sample_rate, 'test_blues_00000')
        visualizer.plot_spectrogram(audio_data, sample_rate, 'test_blues_00000')
        visualizer.visualize(1)
    else:
        print('Skipping Data Visualization')




if __name__ == '__main__':
    main()
