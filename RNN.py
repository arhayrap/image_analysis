from tensorflow import keras


class RNN:

    def __init__(self, x_train, y_train, x_valid, y_valid, test_data):
        self.x_train = x_train
        self.x_val = x_valid
        self.y_train = y_train
        self.y_val = y_valid
        self.test_data = test_data

    @staticmethod
    def create_model():
        model = keras.Sequential()
        model.add(keras.layers.LSTM(input_shape=(None, 450, 450, 3)))
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.Dense(2, activation = "sigmoid"))

        model.compile(optimizer="Adam",
                      loss="mse")
        return model

    def fit(self):
        model = self.create_model()
        model.fit(self.x_train, self.y_train, validation=(self.x_val, self.y_val), batch_size=100, epochs=10, verbose=0)
        print(model.summary())
        return model

    def predict(self, model):
        results = model.predict_classes(self.test_data)
        return results
