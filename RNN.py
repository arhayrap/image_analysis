from tensorflow import keras
import numpy as np


class RNN:

    def __init__(self, x_train, y_train, x_valid, y_valid, test_data):
        self.x_train = x_train
        self.x_val = x_valid
        self.y_train = y_train
        self.y_val = y_valid
        self.test_data = test_data

    def create_model(self):
        # .reshape(450, 450, 1)
        model = keras.Sequential()
        model.add(keras.layers.GRU(150,
                                         return_sequences=False, input_shape=(None, 450)))
        # model.add(keras.layers.Dropout(0.1))

        '''
        model.add(keras.layers.SimpleRNN(150, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.SimpleRNN(150, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        

        model.add(keras.layers.Dense(100,
                                     activation="relu"))
        '''

        model.add(keras.layers.Dense(100,
                                     activation="relu"))
        model.add(keras.layers.Dense(100,
                                     activation="relu"))
        model.add(keras.layers.Dense(2,
                                     activation="sigmoid"))

        model.compile(
                      optimizer="rmsprop",
                      loss="sparse_categorical_crossentropy",
                      metrics=["acc"]
                      )
        return model

    def fit_and_test(self):
        self.y_train = np.expand_dims(self.y_train, axis=-1)
        self.y_val = np.expand_dims(self.y_val, axis=-1)
        print(self.y_train.shape)
        model = self.create_model()
        model.fit(self.x_train,
                  self.y_train,
                  validation_data=(self.x_val, self.y_val),
                  batch_size=20,
                  epochs=8)
        print(model.summary())
        # try:
        #     model.save("./RNN.model")
        # except:
        #     print("Something wrong with the MODEL name or other...")
        model.save("weights")
        results = self.predict(model)
        return results

    def predict(self, model):
        results = model.predict_classes(self.test_data)
        return results
