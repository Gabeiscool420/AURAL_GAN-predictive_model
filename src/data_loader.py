import os
import librosa
import numpy as np
import traceback
from sklearn.model_selection import train_test_split

def load_data():
    try:
        acoustic_guitar_directory = 'acoustic_guitar'
        high_quality_audios = []
        low_quality_audios = []
        max_length = 0

        for guitar_directory in sorted(os.listdir(acoustic_guitar_directory)):
            full_path = os.path.join(acoustic_guitar_directory, guitar_directory)

            if not os.path.isdir(full_path) or guitar_directory in ['guitar_11', 'guitar_12']:
                continue

            high_quality_files = sorted([f for f in os.listdir(full_path) if f.endswith('.wav')])
            low_quality_files = sorted([f for f in os.listdir(os.path.join(full_path, 'low_quality')) if f.endswith('.wav')])

            high_quality_file = high_quality_files[0]
            try:
                high_quality_audio, _ = librosa.load(os.path.join(full_path, high_quality_file), sr=None)
            except Exception as e:
                print(f"Could not load high quality audio file {high_quality_file}: {e}")
                traceback.print_exc()
                continue

            if len(high_quality_audio) > max_length:
                max_length = len(high_quality_audio)

            for low_quality_file in low_quality_files:
                try:
                    low_quality_audio, _ = librosa.load(os.path.join(full_path, 'low_quality', low_quality_file), sr=None)
                except Exception as e:
                    print(f"Could not load low quality audio file {low_quality_file}: {e}")
                    traceback.print_exc()
                    continue

                if len(low_quality_audio) > max_length:
                    max_length = len(low_quality_audio)
                low_quality_audios.append(low_quality_audio)

            high_quality_audios.extend([high_quality_audio] * len(low_quality_files))

        high_quality_spectrograms = []
        low_quality_spectrograms = []
        high_quality_phases = []
        low_quality_phases = []

        for high_quality_audio, low_quality_audio in zip(high_quality_audios, low_quality_audios):
            high_quality_audio = np.pad(high_quality_audio, (0, max_length - len(high_quality_audio)))
            low_quality_audio = np.pad(low_quality_audio, (0, max_length - len(low_quality_audio)))

            high_quality_spectrogram = librosa.stft(high_quality_audio)
            low_quality_spectrogram = librosa.stft(low_quality_audio)

            high_quality_spectrograms.append(np.abs(high_quality_spectrogram))
            low_quality_spectrograms.append(np.abs(low_quality_spectrogram))

            high_quality_phases.append(np.angle(high_quality_spectrogram))
            low_quality_phases.append(np.angle(low_quality_spectrogram))

        high_quality_spectrograms = np.array(high_quality_spectrograms)
        low_quality_spectrograms = np.array(low_quality_spectrograms)
        high_quality_phases = np.array(high_quality_phases)
        low_quality_phases = np.array(low_quality_phases)

        # Print shapes
        print(f"Number of high quality audios: {len(high_quality_audios)}")
        print(f"Shapes of first few high quality audios: {[x.shape for x in high_quality_audios[:5]]}")
        print(f"Number of low quality audios: {len(low_quality_audios)}")
        print(f"Shapes of first few low quality audios: {[x.shape for x in low_quality_audios[:5]]}")
        print(f"High quality spectrograms shape: {high_quality_spectrograms.shape}")
        print(f"Low quality spectrograms shape: {low_quality_spectrograms.shape}")
        print(f"High quality phases shape: {high_quality_phases.shape}")
        print(f"Low quality phases shape: {low_quality_phases.shape}")

        X_train, X_val, Y_train, Y_val = train_test_split(low_quality_spectrograms, high_quality_spectrograms, test_size=0.2, random_state=42)
        _, _, phase_train, phase_val = train_test_split(low_quality_phases, high_quality_phases, test_size=0.2, random_state=42)

        # Print shapes of training and validation sets
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")

        return X_train, X_val, Y_train, Y_val, phase_train, phase_val

    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        traceback.print_exc()

load_data()


