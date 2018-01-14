# -*- coding: utf-8 -*-
"""Read Dataset from MATLAB and write it out in a pickled file"""
import pickle

import numpy as np
import scipy.io as spio

PATH_TO_DATASET = 'D:\Event detection'
DATASET_MATLAB = spio.loadmat(PATH_TO_DATASET + 'Dataset.mat')
NUM_PR, NUM_SAMPLES = np.shape(DATASET_MATLAB['Features'])
DATASET = [[] for i in range(NUM_PR)]
for i in range(0, NUM_PR):
    for j in range(0, NUM_SAMPLES):
        if DATASET_MATLAB['Features'][i, j].size != 0:
            temp = {}
            temp["T"] = DATASET_MATLAB['Features'][i, j][0, 0][0]
            temp["EIHVelocity"] = DATASET_MATLAB['Features'][i, j][0, 0][1]
            temp["HeadVelocity"] = DATASET_MATLAB['Features'][i, j][0, 0][2]
            temp["DirectionSimilarity"] = DATASET_MATLAB['Features'][i, j][0, 0][3]
            temp["EIHAng_AzEl_vel"] = DATASET_MATLAB['Features'][i, j][0, 0][4]
            temp["HAng_AzEl_vel"] = DATASET_MATLAB['Features'][i, j][0, 0][5]
            temp["SceneFrameNo"] = DATASET_MATLAB['Features'][i, j][0, 0][6]
            temp["EyeFrameNo"] = DATASET_MATLAB['Features'][i, j][0, 0][7]
            temp["PathToSceneImages"] = DATASET_MATLAB['Features'][i, j][0, 0][8]
            temp["PathToRightEyeImages"] = DATASET_MATLAB['Features'][i, j][0, 0][9]
            temp["Labels"] = DATASET_MATLAB['Features'][i, j][0, 0][10]
            DATASET[i].append(temp)
pickle.dump(DATASET, open(PATH_TO_DATASET + "Dataset.p", "wb"))