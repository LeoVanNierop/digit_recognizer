from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
import os


class Datacontainer(object):
    def __init__(self, datapath='/home/leo/kaggle/datasets/digit_recognizer/train.csv',
                       testpath='/home/leo/kaggle/datasets/digit_recognizer/test.csv'):
        self.random_state = np.random.RandomState(666)
        df = read_csv(datapath)
        df = shuffle(df, random_state=self.random_state)
        self.targets = df.label.values
        picturecols = [_ for _ in df.columns if _ != 'label']
        self.samples = df.loc[:, picturecols]
        self.normalized_samples = self.samples/255
        self.testsamples = read_csv(testpath).values
        number_of_samples = len(df)
        train_length = int(number_of_samples * 0.6)
        test_length = int(number_of_samples * 0.2)
        self.train_set = self.samples.iloc[:train_length, :]
        self.train_targets = self.targets[:train_length]
        self.test_set = self.samples.iloc[train_length:train_length+test_length, :]
        self.test_targets = self.targets[train_length:train_length+test_length]
        self.validation_set = self.samples.iloc[train_length+test_length:, :]
        self.validation_targets = self.targets[train_length+test_length:]


    def _initialize_one_hot(self):
        self.one_hot_targets = np.zeros((self.targets.shape[0], 10))
        for i, x in enumerate(self.targets):
            self.one_hot_targets[i, x] = 1

    def _get_random_training_set(self, amount, type='one-hot', samples='normalized'):
        if self.random_state == None:
            self.random_state = np.random.RandomState(666)
        sample = self.random_state.choice(len(self.targets), replace=False, size=amount)
        if type == 'one-hot' and samples == 'normalized':
            return self.normalized_samples.loc[sample, :].values, self.one_hot_targets[sample]
        else:
            return self.samples.loc[sample, :].values, self.targets[sample]

    def show_image(self, pic, prediction=None):
        """
        run in ipython with %matplotlib to view sample images
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

        pic = pic.reshape((28,28)).copy()
        if prediction is not None:
            for pos in digitmap[prediction]:
                pic[pos]=255
        plt.imshow(pic, cmap='gray_r')

    def simple_logistic_fit(self):
        self.simple_logistic = LogisticRegression()
        self.simple_logistic.fit(self.samples.values, self.targets)
        joblib.dump(self.simple_logistic, '/home/leo/kaggle/code/digit_recognizer/models/simple_logistic.pk1')

    def tensorflow_logistic_regression(self):
        '''
        implementing the softmax equation:
        y = softmax(Wx + b)
        where for a vector v, softmax(v) is the vector:
        softmax(v)_i = [exp(v_i)]/[sum_j exp(v_j)]

        :return:
        '''
        try:
            self.one_hot_targets
        except:
            self._initialize_one_hot()

        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]) )
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x,W)+b) #predictions
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for i in range(1000):
            if i%100==0:
                print (i)
            batch_xs, batch_ys = self._get_random_training_set(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # get testset and apply
        prediction = tf.argmax(y, 1)
        outputvalues = sess.run(prediction, feed_dict={x: self.testsamples})
        sess.close()
        outdf = DataFrame(outputvalues)
        outdf.index += 1
        outdf.index.name = 'ImageId'
        outdf.columns = ['Label']
        outdf.to_csv('/home/leo/kaggle/code/digit_recognizer/models/tensorflow_logistic2.csv')


        pass


    def simple_svm_fit(self):
        self.simple_svm = SVC()
        self.simple_svm.fit(self.samples.values, self.targets)
        joblib.dump(self.simple_svm, '/home/leo/kaggle/code/digit_recognizer/models/simple_svm.pk1')

    def proper_logistic(self):
        logistic = LogisticRegression()
        grid = [
            {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1],
                'multi_class': ['ovr'],
                'solver': ['liblinear']
            },
            {
                'penalty': ['l2'],
                'C': [0.1, 1],
                'multi_class': ['multinomial'],
                'solver': ['lbfgs']
            }
        ]
        logistic_grid = GridSearchCV(logistic, grid)
        logistic_grid.fit(self.train_set, self.train_targets)
        self.logistic_best_params = logistic_grid.best_params_
        self.logistic_best_score = logistic_grid.best_score_
        self.best_logistic_classifier = logistic_grid.best_estimator_


    def proper_neighbors(self):
        neighbors = KNeighborsClassifier()
        grid = {
            'n_neighbors': [x for x in range(3,7)],
            'weights': ['uniform', 'distance'],
            'p': [1,2],
        }
        neighbors_grid = GridSearchCV(neighbors, grid)
        neighbors_grid.fit(self.train_set, self.train_targets)
        self.neighbors_best_params = neighbors_grid.best_params_
        self.neighbors_best_score = neighbors_grid.best_score_
        self.best_neighbors_classifier = neighbors_grid.best_estimator_

    def find_best_classifier(self):
        print ("starting logistic")
        self.proper_logistic()
        print ("starting neighbors")
        self.proper_neighbors()
        logistic_score = self.best_logistic_classifier.score(self.validation_set, self.validation_targets)
        neighbor_score = self.best_neighbors_classifier.score(self.validation_set, self.validation_targets)
        scoring = {
            'logistic': (logistic_score, self.best_logistic_classifier),
            'neighbor': (neighbor_score, self.best_neighbors_classifier)
        }
        maxkey = max(scoring.items(), key=lambda x: x[1][0])[0]
        for key in scoring:
            save_path = os.path.join('/home/leo/kaggle/code/digit_recognizer/models', key)
            save_path = os.path.join(save_path, 'model.pkl')
            joblib.dump(scoring[key][1], save_path)
        print (maxkey)

    def load_classifiers(self):
        self.best_neighbors_classifier = joblib.load('/home/leo/kaggle/code/digit_recognizer/models/neighbor/model.pkl')
        self.best_logistic_classifier = joblib.load('/home/leo/kaggle/code/digit_recognizer/models/logistic/model.pkl')
        self.pca_logistic = joblib.load('/home/leo/kaggle/code/digit_recognizer/models/pca_logistic/model.pkl')


    def show_logistic_prediction(self):
        choice = self.random_state.choice(range(len(self.test_set)))
        input, target = self.test_set.iloc[choice, :].values.reshape(1,-1), self.test_targets[choice]
        prediction = self.best_logistic_classifier.predict(input)
        self.show_image(input, prediction[0])

    def show_neighbors_prediction(self):
        choice = self.random_state.choice(range(len(self.test_set)))
        input, target = self.test_set.iloc[choice, :].values.reshape(1,-1), self.test_targets[choice]
        prediction = self.best_neighbors_classifier.predict(input)
        self.show_image(input, prediction[0])

    def pipeline_example(self):
        pca = PCA(n_components=20)
        logistic = LogisticRegression(penalty='l2', C=1, multi_class='multinomial', solver='lbfgs')
        self.pca_logistic = Pipeline([('pca', pca), ('logistic', logistic)])
        self.pca_logistic.fit(self.train_set, self.train_targets)
        joblib.dump(self.pca_logistic, '/home/leo/kaggle/code/digit_recognizer/models/pca_logistic/model.pkl')
        
    def show_pca_logistic_prediction(self):
        choice = self.random_state.choice(range(len(self.test_set)))
        input, target = self.test_set.iloc[choice, :].values.reshape(1,-1), self.test_targets[choice]
        prediction = self.pca_logistic.predict(input)
        self.show_image(input, prediction[0])

dat = Datacontainer()
dat.load_classifiers()




