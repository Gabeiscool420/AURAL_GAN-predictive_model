from tensorflow import keras
import numpy as np
import traceback
from tensorflow.keras.callbacks import ModelCheckpoint


class Discriminator:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        try:
            model = keras.models.Sequential()
            model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(None, None, 1)))
            model.add(keras.layers.LeakyReLU(alpha=0.2))
            model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LeakyReLU(alpha=0.2))
            model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LeakyReLU(alpha=0.2))
            model.add(keras.layers.GlobalAveragePooling2D())
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam')
            return model
        except Exception as e:
            print(f"An error occurred building discriminator: {e}")
            traceback.print_exc()

    def train(self, real_spectrograms, fake_spectrograms, epochs, batch_size):
        try:
            if fake_spectrograms is None:
                print("No fake spectrograms were generated. Aborting training.")
                return

            X_train = np.concatenate((real_spectrograms, real_spectrograms), axis=0)
            y_train = np.zeros((2 * batch_size,))
            y_train[:batch_size] = 0.9


            # Change the checkpoint to save the entire model, not just weights
            checkpoint = ModelCheckpoint('discriminator-{epoch:03d}.h5', verbose=1, monitor='val_loss',
                                         save_best=True, mode='auto', save_weights=False)

            tensorboard = TensorBoard(log_dir='./logs_discriminator', histogram_freq=0, write_graph=True,
                                      write_images=True)
            history = self.model.fit(X_train, y_train,
                                     validation_data=(fake_spectrograms, np.ones((fake_spectrograms.shape[0],))),
                                     epochs=epochs,
                                     batch_size=batch_size, callbacks=[checkpoint, tensorboard], verbose=1)

            # Save and print loss values
            loss_values = history.history['loss']
            print("Loss values: ", loss_values)
            np.save('discriminator_loss_values.npy', loss_values)

            # At the end of training, save the entire model
            self.model.save('discriminator_final.h5')

        except Exception as e:
            print(f"An error occurred during training: {e}")
            traceback.print_exc()

        except Exception as e:
            print(f"An error occurred during training: {e}")
            traceback.print_exc()
            return

    def discriminate(self, spectrogram):
        try:
            return self.model.predict(spectrogram[np.newaxis, ...])
        except Exception as e:
            print(f"An error occurred during discrimination: {e}")
            traceback.print_exc()
            return None
