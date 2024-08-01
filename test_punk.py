import librosa
import matplotlib.pyplot as plt

# Load an audio file
test_audio_file = 'genres/blues/blues.00000.wav'


# sr=None preserves the original sampling rate
y, sr = librosa.load(test_audio_file, sr=None)
print(f"TEST Audio data shape: {y.shape}")
print(f"TEST Sampling rate: {sr}")

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Display the MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
plt.close()