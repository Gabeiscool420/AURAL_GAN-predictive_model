import traceback
import generator
from generator import Generator
from discriminator import Discriminator
from data_loader import load_data
import numpy as np
import librosa


def main():
    try:
        X_train, x_test, y_train, y_test, phase_train, phase_test = load_data()
        generator = Generator()
        discriminator = Discriminator()
        generator.train(X_train, y_train, epochs=10, batch_size=8)
        fake_spectrograms = generator.generate(x_test)
        discriminator.train(y_test, fake_spectrograms, epochs=10, batch_size=8)
        test_spectrogram = x_test[np.random.choice(len(x_test))]
        generated_spectrogram = generator.generate(test_spectrogram)

        if generated_spectrogram is not None:
            try:
                classification = discriminator.discriminate(generated_spectrogram)
                print(f"The generated spectrogram was classified as {'real' if classification > 0.5 else 'fake'}.")
                test_phase = phase_test[np.random.choice(len(phase_test))]
                generated_audio = generator.spectrogram_to_audio(generated_spectrogram, test_phase, 'output.wav')
                print(f"Generated audio: {generated_audio}")
            except Exception as e:
                print(f"couldnt discriminate: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {x_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Phase_train shape: {phase_train.shape}")
        print(f"Phase_test shape: {phase_test.shape}")
        print(f"Fake_spectrograms shape: {fake_spectrograms.shape}")
        print(f"Test_spectrogram shape: {test_spectrogram.shape}")
        print(f"Generated_spectrogram shape: {generated_spectrogram.shape}")
        print(f"Generated_audio shape: {generated_audio.shape}")

def enhance_audio(file_path, generator):
    try:
        # Load the low-quality audio from the file path
        low_quality_audio, _ = librosa.load(file_path, sr=None)

        # Convert the audio to a spectrogram
        low_quality_spectrogram = np.abs(librosa.stft(low_quality_audio))

        # Use the trained generator to enhance the spectrogram
        enhanced_spectrogram = generator.generate(low_quality_spectrogram)

        # Retrieve the phase from the low-quality spectrogram
        phase = np.angle(librosa.stft(low_quality_audio))

        # Convert the enhanced spectrogram back to audio
        enhanced_audio = generator.spectrogram_to_audio(enhanced_spectrogram, phase, file_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    main()
    file_path = input("Enter the path to a low-quality audio file to enhance: ")
    enhance_audio(file_path, generator)