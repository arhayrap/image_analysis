import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os


def get_data():
    # data = [{"image": [], "label": []},
    #         {"image": [], "label": []},
    #         {"image": [], "label": []}]

    # start_end = [(0, 2),
    #              (2, 4),
    #              (4, None)]

    data = [{"image": [], "label": []}]
    start_end = [(0, 1)]

    for u in range(len(start_end)):
        path = "../../samples_Aram"
        m_path = path
        for i in os.listdir(m_path):
            m_path = path + "/" + i
            if i == "For_Train_Valid":
                for j in os.listdir(m_path)[start_end[u][0]: start_end[u][1]]:
                    print(j)
                    if os.path.isdir(m_path + "/" + j):
                        n_path = m_path + "/" + j
                        for k in os.listdir(n_path):
                            if k == "cop":
                                k_path = n_path + "/" + k
                                for t in os.listdir(k_path):
                                    try:
                                        data[u]["image"].append(np.array(cv2.resize(cv2.cvtColor(cv2.imread(k_path + "/" + t),
                                                                                        cv2.COLOR_BGR2GRAY),
                                                                           (450, 450))))
                                        data[u]["label"].append(0)
                                    except:
                                        print(k_path + "/" + t)
                            elif k == "gen":
                                k_path = n_path + "/" + k
                                for t in os.listdir(k_path):
                                    try:
                                        data[u]["image"].append(np.array(cv2.resize(cv2.cvtColor(cv2.imread(k_path + "/" + t),
                                                                                        cv2.COLOR_BGR2GRAY),
                                                                           (450, 450))))
                                        data[u]["label"].append(1)
                                    except:
                                        print(k_path + "/" + t)
                            else:
                                continue
                    else:
                        continue
            else:
                continue

    tdata = []
    # data = pd.DataFrame({"image": data[0]["image"] + data[1]["image"] + data[2]["image"],
    #                     "label": data[0]["label"] + data[1]["label"] + data[2]["label"]})
    data = pd.DataFrame(data[0])
    path = "../../samples_Aram"
    m_path = path
    for i in os.listdir(m_path):
        m_path = path + "/" + i
        if i == "For_Tests":
            for j in os.listdir(m_path):
                n_path = m_path + "/" + j
                for k in os.listdir(n_path):
                    tdata.append(np.array(cv2.resize(cv2.cvtColor(cv2.imread(n_path + "/" + k),
                                                                  cv2.COLOR_BGR2GRAY),
                                                     (450, 450))))
        else:
            continue

    t_data = np.array(tdata)

    return data, t_data


