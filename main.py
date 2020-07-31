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

    train_valid_data = shuffle(get_data()[0])
    test_data = shuffle(get_data()[1])
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_valid_data["image"], train_valid_data["label"], test_size=0.25)
    model = RNN(X_train, X_valid, Y_train, Y_valid, test_data.create_model())
    # plt.imshow(np.array(np.array(X_train[0])))
    # print(Y_valid.head())
    # print(test_data.shape)


if __name__ == "__main__":
    main()
