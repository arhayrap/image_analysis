import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from read_images import get_data
from RNN import RNN


def main():
    '''
    tr_paths = ["./Datasets/Training_Data1.csv",
                "./Datasets/Training_Data2.csv",
                "./Datasets/Training_Data3.csv"]

    ts_paths = ["./Datasets/Testing_Data.csv"]
    train_valid_data = pd.DataFrame({"image": [], "label": []})
    test_data = pd.DataFrame({"image": []})
    for i in tr_paths:
        train_valid_data = pd.concat([train_valid_data, pd.read_csv(i)], axis=0, sort=False)
    for i in ts_paths:
        test_data = pd.concat([test_data, pd.read_csv(i)], axis=0, sort=False)
    '''
    data = get_data()
    train_valid_data = shuffle(data[0])
    test_data = shuffle(data[1])
    print("Data has been collected!")
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(train_valid_data["image"]),
                                                          np.array(train_valid_data["label"]),
                                                          test_size=0.25)
    # x_train = x_train.to_numpy()
    # x_valid = x_valid.to_numpy()
    # y_train = y_train.to_numpy()
    # y_valid = y_valid.to_numpy()
    print(x_train)
    print(x_train.shape, x_train[0].shape)
    model = RNN(x_train,  y_train, x_valid, y_valid, test_data)
    print("Model training process")
    results = model.fit_and_test()
    print("The results are ready!")
    return results


if __name__ == "__main__":
    print(main())
