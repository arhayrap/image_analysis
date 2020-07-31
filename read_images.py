import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os


def get_data():
    data = [{"image": [], "label": []},
            {"image": [], "label": []},
            {"image": [], "label": []}]

    start_end = [(0, 4),
                 (4, 8),
                 (8, 12)]


    for u in range(len(start_end)):
        path = "../samples_Aram"
        m_path = path
        for i in os.listdir(m_path):
            m_path = path + "/" + i
            if i == "For_Train_Valid":
                for j in os.listdir(m_path)[start_end[u][0]: start_end[u][1]]:
                    print(j)
                    if os.path.isdir(m_path + "/" + j):
                        n_path = m_path + "/" + j
                        for k in os.listdir(n_path + "/" + j):
                            if k == "cop":
                                k_path = n_path + "/" + j + "/" + k
                                for t in os.listdir(k_path):
                                    data[u]["image"].append(cv2.resize(cv2.imread(k_path + "/" + t), (450, 450)))
                                    data[u]["label"].append(0)
                            elif k == "gen":
                                k_path = n_path + "/" + j + "/" + k
                                for t in os.listdir(k_path):
                                    data[u]["image"].append(cv2.resize(cv2.imread(k_path + "/" + t), (450, 450)))
                                    data[u]["label"].append(1)
                            else:
                                continue
                    else:
                        continue
            else:
                continue

    tdata = {"image": []}
    data = pd.DataFrame({"image": data[0]["image"] + data[1]["image"] + data[2]["image"],
                        "label": data[0]["label"] + data[1]["label"] + data[2]["label"]})
    path = "../samples_Aram"
    m_path = path
    for i in os.listdir(m_path):
        m_path = path + "/" + i
        if i == "For_Tests":
            for j in os.listdir(m_path):
                n_path = m_path + "/" + j
                for k in os.listdir(n_path):
                    tdata["image"].append(cv2.imread(n_path + "/" + k))
        else:
            continue

    t_data = pd.DataFrame(tdata)

    return data, t_data


