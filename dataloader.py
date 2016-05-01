from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np


class Datacontainer(object):
    def __init__(self, datapath='/home/leo/kaggle/datasets/digit_recognizer/train.csv'):
        df = read_csv(datapath)
        self.targets = df.label.values
        picturecols = [_ for _ in df.columns if _ != 'label']
        self.samples = df.loc[:, picturecols]

    def show_image(self, index, prediction=None):
        """
        run in ipython with %mathplotlib to view sample images
        :param index:
        :return:
        """
        digitmap = {
            0: [(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (4,1), (0,2), (1,2), (2,2), (3,2), (4,2)],
            1: [(0,2), (1,2), (2,2), (3,2), (4,2)],
            2: [(0,0), (0,1), (0,2), (1,2), (2,0), (2,1), (2,2), (3,0), (4,0), (4,1), (4,2)],
            3: [(0,0), (0,1), (0,2), (1,2), (2,0), (2,1), (2,2), (3,2), (4,0), (4,1), (4,2)],
            4: [(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (3,2), (4,2)],
            5: [(0,0), (0,1), (0,2), (1,0), (2,0), (2,1), (2,2), (3,2), (4,0), (4,1), (4,2)],
            6: [(0,0), (0,1), (0,2), (1,0), (2,0), (2,1), (2,2), (3,0), (3,2), (4,0), (4,1), (4,2)],
            7: [(0,0), (0,1), (0,2), (1,2), (2,2), (3,2), (4,2)],
            8: [(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (4,1), (0,2), (1,2), (2,2), (3,2), (4,2), (2,1)],
            9: [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2), (3,2), (4,0), (4,1), (4,2)]
        }

        pic = self.samples.loc[index, :].values.reshape((28,28)).copy()
        if prediction is not None:
            for pos in digitmap[prediction]:
                pic[pos]=255
        plt.imshow(pic, cmap='gray_r')

    def simple_logistic_fit(self):
        self.simple_logistic = LogisticRegression()
        self.simple_logistic.fit(self.samples.values, self.targets)
        joblib.dump(self.simple_logistic, '/home/leo/kaggle/code/digit_recognizer/models/simple_logistic.pk1')


    def simple_svm_fit(self):
        self.simple_svm = SVC()
        self.simple_svm.fit(self.samples.values, self.targets)
        joblib.dump(self.simple_svm, '/home/leo/kaggle/code/digit_recognizer/models/simple_svm.pk1')