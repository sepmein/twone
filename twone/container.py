import random

import numpy as np
import pandas as pd
from sklearn import preprocessing


class Container:
    """
        Basic container for the neural networks
    """

    def __init__(self,
                 data_frame,
                 training_set_split_ratio=0.7,
                 cross_validation_set_split_ratio=0.2,
                 test_set_split_ratio=0.1):
        """

        :param data_frame: pandas data frame object
        """
        self.data = data_frame
        self.scale = preprocessing.StandardScaler()

        # data normalizer
        self.normalizer = None

        # take control over randomness
        self.__random_seed__ = random.seed(a=10)

        # init feature data and target data
        self.feature_tags = []
        self.target_tags = []
        self.__feature_data__ = None
        self.__target_data__ = None

        # create train/cv/test dataSet
        self.training_set_split_ratio = training_set_split_ratio
        self.cross_validation_set_split_ratio = cross_validation_set_split_ratio
        self.test_set_split_ratio = test_set_split_ratio

    def detect_null(self):
        """
        detect any null data in the data_frame
        :return:
        """
        # TODO: To make this fn fitting to different ranks
        return self.data.isnull().any().any()

    def interpolate(self, data=None, method='linear'):
        """
        interpolate
        :param data:
        :param method:
        :return:
        """
        if data is None:
            data = self.data
        self.data = data.interpolate(method=method)
        self.data = self.data.dropna()
        return self

    def set_feature_tags(self, feature_tags):
        """

        :param feature_tags:
        :return:
        """
        self.feature_tags = feature_tags
        return self

    def append_feature_tags(self, feature_tags):
        """
        append 'feature-tags' to self.feature_tags
        :param feature_tags:
        :return:
        """
        for tag in feature_tags:
            self.feature_tags.append(tag)
        return self

    def set_target_tags(self, target_tags):
        """

        :param target_tags:
        :return:
        """
        self.target_tags = target_tags
        return self

    def add_feature_tag(self, tag):
        """

        :param tag:
        :return:
        """
        self.feature_tags.append(tag)
        return self

    def add_target_tag(self, tag):
        """

        :param tag:
        :return:
        """
        self.target_tags.append(tag)
        return self

    def fit(self, data=None):
        """

        :param data:
        :return:
        """
        if data is None:
            data = self.data
        self.normalizer = self.scale.fit(data)
        return self

    def normalize(self, data=None):
        """
        Normalized data using predefined scale
        should always normalized feature data, not target data
        :param data:
        :return:
        """
        if data is None:
            data = self.data
        return self.normalizer.transform(data)

    def compute_feature_data(self, shuffle=False):
        """
        compute feature data based on feature tags
        :return:
        """
        if self.feature_tags is None:
            raise Exception('Feature tags not set, can\'t get feature data')
        result = self.data[self.feature_tags]
        # fit and transform feature data
        self.fit(result)
        normalized_array = self.normalize(result)
        constructed_df = pd.DataFrame(normalized_array, columns=self.feature_tags)
        self.__feature_data__ = constructed_df
        return self

    def compute_target_data(self):
        """
        compute target data based on target tags
        :return: self
        """
        if self.target_tags is None:
            raise Exception('Target tags not set, can\'t get target data')
        self.__target_data__ = self.data[self.target_tags]
        return self


class DNNContainer(Container):
    """
        Container for densely connected Neural Network
    """

    def __init__(self,
                 data_frame):
        """

        :param data_frame:
        """
        Container.__init__(self, data_frame)

        # create train/cv/test dataSet
        self.__training_set_mask__ = None
        self.__cross_validation_set_mask__ = None
        self.__test_set_mask__ = None

    def gen_mask(self, days=0):
        # define pseudo mask for generating class
        mask = np.random.rand(len(self.data) - days)
        self.__training_set_mask__ = mask <= self.training_set_split_ratio
        self.__cross_validation_set_mask__ = (mask > self.training_set_split_ratio) & \
                                             (mask < (self.training_set_split_ratio
                                                      + self.cross_validation_set_split_ratio))
        self.__test_set_mask__ = mask >= (self.training_set_split_ratio + self.cross_validation_set_split_ratio)

    def compute_feature_data(self, shuffle=False):
        """

        :return:
        """
        if self.feature_tags is None:
            raise Exception('Feature tags not set, can\'t get feature data')
        result = self.data[self.feature_tags]
        # fit and transform feature data
        self.fit(result)
        normalized_array = self.normalize(result)
        constructed_df = pd.DataFrame(normalized_array, columns=self.feature_tags)
        self.__feature_data__ = constructed_df
        return self

    def compute_target_data(self):
        """

        :return:
        """
        if self.target_tags is None:
            raise Exception('Target tags not set, can\'t get target data')
        self.__target_data__ = self.data[self.target_tags]
        return self

    def get_last_data(self, days=30):
        gen_labels = []
        gen_data = []
        data = self.__feature_data__
        labels = self.feature_tags
        num_data = data.shape[0]
        starts_at = days
        num_feature_labels = days * len(self.feature_tags)

        for label in self.feature_tags:
            for j in range(days):
                gen_labels.append(label + '_' + str(j + 1))

        for k in range(starts_at, num_data):
            _from = k - days
            _to = k
            days_back_data = data[_from: _to]
            selected_day_back_data = days_back_data.loc[:, labels[0]:labels[-1]]
            selected_day_back_data_np = selected_day_back_data.values
            reshaped = np.reshape(selected_day_back_data_np.T,
                                  (1, num_feature_labels))
            gen_data.append(reshaped)
        stacked = np.vstack(row for row in gen_data)
        data_frame = pd.DataFrame(data=stacked, columns=gen_labels)
        self.__feature_data__ = data_frame
        self.__target_data__ = self.__target_data__[days:]
        return self

    def get_training_features(self):
        """

        :return:
        """
        return self.__feature_data__[self.__training_set_mask__]

    def get_training_targets(self):
        """

        :return:
        """
        return self.__target_data__[self.__training_set_mask__]

    def get_cross_validation_features(self):
        """

        :return:
        """
        return self.__feature_data__[self.__cross_validation_set_mask__]

    def get_cross_validation_targets(self):
        """

        :return:
        """
        return self.__target_data__[self.__cross_validation_set_mask__]

    def get_test_features(self):
        """

        :return:
        """
        return self.__feature_data__[self.__test_set_mask__]

    def get_test_targets(self):
        """

        :return:
        """
        return self.__target_data__[self.__test_set_mask__]


class RNNContainer(Container):
    """
        A container pre-process recurrent data, and generate the data conform to Tensorflow style.
        The container should reshape data into the shape of (batch size, times steps, feature length), one of these unit
        is one epoch. Container should iteratively output lots of epochs.

        The length of the total sequence data is total_length. It should be spliced to lots of batches.
        e.g.:

        From:
        | <-                                       total-length                                                    -> |
        ---------------------------------------------------------------------------------------------------------------

        To:
        |        batches      |
        -----------------------
        -----------------------

                ........
        -----------------------                 num_batches
        -----------------------
        -----------------------
        -----------------------
        -----------------------

        After above procedure, above data should be spliced to different epoches.

        From:
        |        batches      |
        -----------------------
        -----------------------

                ........
        -----------------------
        -----------------------
        -----------------------
        -----------------------
        -----------------------

        To:
        | ep1 | | ep2 | | ep3 |  ...   | epn |
        ------- ------- ------         -------
        ------- ------- ------         -------

                    .............
        ------- ------- ------         -------
        ------- ------- ------         -------
        ------- ------- ------         -------
        ------- ------- ------         -------
        ------- ------- ------         -------

        Look at just one epoch:

        | epi |    | num of time steps = 7 |
        -------    -   -   -   -   -   -   -
                   1   2   3   4   5   6   7

        The num_steps of max_time value is one hyper-parameter that should been tuned.
        In theory, the gradient of recurrent neural network could flow back to a **really** long distance. But in pract-
        ice, the back propagation through time(BPTT) algorithm can't flow back too long due to the calculation resources.
        So Truncated back propagation through time is normally used in practice.
    """

    def __init__(self,
                 data_frame):
        """

        :param data_frame:
        """
        Container.__init__(self, data_frame)
        self._total_length = self.data.shape[0]

        # training, cv and test data
        self.__training_features__ = None
        self.__training_targets__ = None
        self.__cv_features__ = None
        self.__cv_targets__ = None
        self.__test_features__ = None
        self.__test_targets__ = None

    @property
    def num_features(self):
        """

        :return: type Int, number of features of target
        """
        if self.feature_tags is None:
            return None
        else:
            return len(self.feature_tags)

    @property
    def num_targets(self):
        """

        :return: type Int, number of features of target
        """
        if self.target_tags is None:
            return None
        else:
            return len(self.target_tags)

    def set_target_tags(self, target_tags, shift=-1):
        """
        set target tags fn for rnn model.
        accept target_tags as a string of list, and a single shift value which indicates how much steps feature data
        will shift.
        :param target_tags: list.
        :param shift: int
        :return: self
        """
        true_target_tags = [tag + '_target' for tag in target_tags]
        # copy the target column to targetName + '_target' column, so that the original target tag could be used
        # as an feature
        for tag in target_tags:
            self.data[tag + '_target'] = self.data[tag]
        # for every target_tag in target_tags, shift back by "shift" parameter
        for tag in true_target_tags:
            self.data[tag] = self.data[tag].shift(shift)
        # For seq labeling problem, remove the row that included with NAN results
        self.data = self.data[0:shift]
        # set target tags and feature tags
        # because target tags could be used as feature tags
        self.append_feature_tags(target_tags)
        self.target_tags = true_target_tags
        return self

    def gen_batch(self, batch=5, time_steps=128):
        """
        reshape feature data into (batch, max_time, num_features) for tf.nn.dynamic_rnn function
        check tensorflow documentations for explanation
        current version is 1.4
        :param batch:
        :param time_steps:
        :return:
        """
        if self.feature_tags is None:
            raise Exception('Feature tags not set, can\'t get feature data')
        features = self.data[self.feature_tags]
        # fit and transform feature data
        self.fit(features)
        normalized_array = self.normalize(features)
        # calculate dims for reshape
        dim_0 = batch
        dim_1 = batch_partition_size = self._total_length // batch
        # reshape features
        features_reshaped_by_batch = np.reshape(normalized_array, [dim_0, dim_1, self.num_features])
        # reshape targets
        targets_array = self.data[self.target_tags].values
        targets_reshaped_by_batch = np.reshape(targets_array, [dim_0, dim_1, self.num_targets])
        # calculate epochs
        epochs = batch_partition_size // time_steps
        # print epoch length for debug
        # TODO: remove it later
        print(epochs)

        # compute test data length
        test_data_length = epochs * self.test_set_split_ratio
        if test_data_length <= 1:
            test_data_length = 1
        else:
            test_data_length = round(test_data_length)

        # compute cross validation data length
        cv_data_length = epochs * self.cross_validation_set_split_ratio
        if cv_data_length <= 1:
            cv_data_length = 1
        else:
            cv_data_length = round(cv_data_length)

        # compute training data length by epochs, cv_data_length and test_data_length
        training_data_length = epochs - cv_data_length - test_data_length

        if training_data_length < 1:
            raise Exception('[twone Exception]: rnn.gen_batch | training data epoch is less than 1, please consider' +
                            'lower down the batch size or increase total data length')

        training_features = []
        training_targets = []
        cv_features = []
        cv_targets = []
        test_features = []
        test_targets = []
        for i in range(epochs):
            current_feature = features_reshaped_by_batch[:, time_steps * i: time_steps * (i + 1):]
            current_target = targets_reshaped_by_batch[:, time_steps * i: time_steps * (i + 1):]
            if i < training_data_length:
                training_features.append(current_feature)
                training_targets.append(current_target)
            elif i < training_data_length + cv_data_length:
                cv_features.append(current_feature)
                cv_targets.append(current_target)
            else:
                test_features.append(current_feature)
                test_targets.append(current_target)

        self.__training_features__ = training_features
        self.__training_targets__ = training_targets
        self.__cv_features__ = cv_features
        self.__cv_targets__ = cv_targets
        self.__test_features__ = test_features
        self.__test_targets__ = test_targets
        return self

    def get_training_features(self):

        """

        :return:
        """
        for feature in self.__training_targets__:
            yield feature

    def get_training_targets(self):

        """

        :return:
        """
        for target in self.__training_targets__:
            yield target

    def get_cross_validation_features(self):

        """

        :return:
        """
        for feature in self.__cv_targets__:
            yield feature

    def get_cross_validation_targets(self):

        """

        :return:
        """
        for target in self.__cv_targets__:
            yield target

    def get_test_features(self):

        """

        :return:
        """
        for feature in self.__test_features__:
            yield feature

    def get_test_targets(self):

        """

        :return:
        """
        for target in self.__test_targets__:
            yield target

    def dealing_with_missing_data(self):
        """
        High level method combined by other methods to deal with missing data in time series.
        :return:
        """
        self.interpolate()
        # after interpolation, if there are any data in the first row or last row is missing,
        # The simple way of dealing such data is to drop it.
        self.data.dropna()
        self.fit()
        self.normalize()
