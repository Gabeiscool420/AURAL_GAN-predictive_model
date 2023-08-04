from tensorflow import keras
import tensorflow as tf
import numpy as np
import librosa
import traceback
from tensorflow.keras.callbacks import ModelCheckpoint
import soundfile as sf
from tensorflow.keras.callbacks import TensorBoard


class CustomCropping2D(keras.layers.Layer):
    def call(self, inputs):
        try:
            x, ref = inputs  # input is a list of two tensors: the tensor to crop and the reference tensor
            crop_height = tf.shape(ref)[1] - tf.shape(x)[1]
            crop_width = tf.shape(ref)[2] - tf.shape(x)[2]
            return x[:, :crop_height, :crop_width, :]
        except Exception as e:
            print(f"An error occurred 2d cropping: {e}")
            traceback.print_exc()

class Generator:
    def __init__(self):
        try:
            self.model = self.build_model()
        except Exception as e:
            print(f"could not init self: {e}")
            traceback.print_exc()

    def build_model(self):
        try:
            input_layer = keras.layers.Input(shape=(None, None, 1))
            down1 = keras.layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2))(input_layer)
            down1 = keras.layers.BatchNormalization()(down1)
            down1 = keras.layers.Activation('relu')(down1)
            center = keras.layers.Conv2D(128, (3, 3), padding='same')(down1)
            center = keras.layers.BatchNormalization()(center)
            center = keras.layers.Activation('relu')(center)
            up1 = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(center)
            up1 = CustomCropping2D()([up1, down1])
            up1 = keras.layers.concatenate([down1, up1], axis=3)
            up1 = keras.layers.BatchNormalization()(up1)
            up1 = keras.layers.Activation('relu')(up1)
            up2 = keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(up1)
            up2 = keras.layers.Activation('tanh')(up2)
            up2 = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(up2)
            model = keras.models.Model(inputs=input_layer, outputs=up2)
            model.compile(loss='mean_squared_error', optimizer='adam')

            # Add print statements here
            print(f"Input shape: {input_layer.shape}")
            print(f"Down1 shape: {down1.shape}")
            print(f"Center shape: {center.shape}")
            print(f"Up1 shape: {up1.shape}")
            print(f"Output shape: {up2.shape}")

            return model
        except Exception as e:
            print(f"An error occurred during building generator: {e}")
            traceback.print_exc()

    def train(self, X_train, y_train, epochs, batch_size):
        try:
            checkpoint = ModelCheckpoint('generator-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best=True,
                                         mode='auto', save_weights=True, save_freq='epoch')
            tensorboard = TensorBoard(log_dir='./logs_generator', histogram_freq=0, write_graph=True, write_images=True)
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                     callbacks=[checkpoint, tensorboard],
                                     verbose=1)

            # Save and print loss values
            loss_values = history.history['loss']
            print("Loss values: ", loss_values)
            np.save('generator_loss_values.npy', loss_values)

            # Add print statement here
            print(f"Training on data with shape {X_train.shape} for {epochs} epochs with batch size {batch_size}")

        except Exception as e:
            print(f"An error occurred during training: {e}")
            traceback.print_exc()

    def generate(self, input_spectrogram):
        try:
            # Add a dimension for the batch, but not for the channel
            input_spectrogram = np.expand_dims(input_spectrogram, axis=0)
            generated_spectrogram = self.model.predict(input_spectrogram)
            return generated_spectrogram[0, ..., 0]
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            return None

    def spectrogram_to_audio(self, magnitude, phase, filename):
        try:
            # Combine magnitude and phase information to a complex-valued spectrogram
            spectrogram = magnitude * np.exp(1j * phase)

            # Apply the inverse STFT to get the time-domain audio signal
            audio = librosa.istft(spectrogram)

            print(f"Magnitude shape: {magnitude.shape}")
            print(f"Phase shape: {phase.shape}")
            print(f"Spectrogram shape: {spectrogram.shape}")
            print(f"Audio shape: {audio.shape}")

            # Write the audio data to a .wav file
            sf.write(filename, audio, 44100)  # Assuming a sample rate of 44100 Hz

            print(f"Audio data written to: {filename}")

            return audio
        except Exception as e:
            print(f"An error occurred while attempting to write the Audio...: {e}")
            traceback.print_exc()
