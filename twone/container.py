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
            data = self.data[self.feature_tags].values
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
            data = self.data[self.feature_tags].values
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

        # training, cv and test data
        self.__training_features__ = None
        self.__training_targets__ = None
        self.__cv_features__ = None
        self.__cv_targets__ = None
        self.__test_features__ = None
        self.__test_targets__ = None
        self.__batch__ = None
        self.__random__ = None

        self.__training_pointer__ = 0
        self.__cv_pointer__ = 0
        self.__test_pointer__ = 0

    @property
    def _total_length(self):
        """
        calculate total length of the container
        :return: int
        """
        return self.data.shape[0]

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

    def gen_batch(self,
                  batch=5,
                  random_batch=True,
                  time_steps=128,
                  shuffle=True,
                  truncate_from_head=True
                  ):
        """
        reshape feature data into (batch, max_time, num_features) for tf.nn.dynamic_rnn function
        check tensorflow documentations for explanation
        current version is 1.4
        :param batch:
        :param random_batch: Bool
        :param time_steps:
        :param shuffle:
        :param truncate_from_head:
        :return:
        """
        # === step 0 ===
        # Get all data as numpy array
        # TODO, what if data is too large to fit in the memory?
        # fit and transform feature data
        self.fit()
        normalized_features = self.normalize()
        self.data.loc[:, self.feature_tags] = normalized_features
        all_data = self.data.values

        # === step 1 ===
        # Calculate if the total length could been divided by time_steps exactly.
        # Get number of sequences, remainder (if not divisible)
        remainder = 0
        sequences = self._total_length // time_steps
        if self._total_length % time_steps > 0:
            divisible = False
            remainder = self._total_length - time_steps * sequences
        else:
            divisible = True

        # === step 2 ===
        # Truncate head or tail by truncate_from_head argument
        if not divisible:
            if truncate_from_head:
                truncated = all_data[remainder:, :]
            else:
                truncated = all_data[:self._total_length - remainder, :]
        else:
            truncated = all_data

        # === step 3 ===
        # Reshape data into (sequences, time_steps, num_features + num_targets)
        # Shuffle all data based on the first dim of reshaped data
        # So that the sequence is shuffled and the time_steps is kept.
        all_data_reshaped = np.reshape(truncated, (sequences, time_steps, self.num_features + self.num_targets))
        if shuffle:
            np.random.shuffle(all_data_reshaped)

        # === step 4 ===
        # Finally get features and targets here
        features = all_data_reshaped[:, :, :self.num_features]
        targets = all_data_reshaped[:, :, self.num_features:]

        # === step 5 ===
        # compute test data length
        test_data_length = sequences * self.test_set_split_ratio
        test_data_length = round(test_data_length)

        # compute cross validation data length
        cv_data_length = sequences * self.cross_validation_set_split_ratio
        cv_data_length = round(cv_data_length)

        # compute training data length by epochs, cv_data_length and test_data_length
        training_data_length = sequences - cv_data_length - test_data_length

        # === step 6 ===
        # copy to self.variables
        self.__training_features__ = features[:training_data_length, :, :]
        self.__training_targets__ = targets[:training_data_length, :, :]
        self.__cv_features__ = features[training_data_length:training_data_length + cv_data_length, :, :]
        self.__cv_targets__ = targets[training_data_length: training_data_length + cv_data_length, :, :]
        self.__test_features__ = features[training_data_length + cv_data_length:, :, :]
        self.__test_targets__ = targets[training_data_length + cv_data_length, :, :]
        self.__batch__ = batch
        self.__random__ = random_batch
        return self

    def get_training_features_and_targets(self):

        """

        :return:
        """
        training_sequences_length = self.__training_features__.shape[0]
        if self.__batch__ > training_sequences_length:
            raise Exception('Batch size is too big, consider low down batch size')
        while True:
            if self.__random__:
                index = np.random.randint(
                    low=0,
                    high=training_sequences_length,
                    size=self.__batch__)
                yield (self.__training_features__[index], self.__training_targets__[index])
            else:
                start = self.__training_pointer__ * self.__batch__
                end = (self.__training_pointer__ + 1) * self.__batch__
                # Pointer greater than the total length of the training data
                # 如果一旦开始或者终止的index越界，那么就归零，并且返回最上面的那一串数据
                # 这样的坏处是，最后那一小部分除不净的数据永远也用不到了，就好像被删除了
                if start > training_sequences_length or end > training_sequences_length:
                    self.__training_pointer__ = 0
                    yield (self.__training_features__[0: self.__batch__], self.__training_targets__[0: self.__batch__])
                else:
                    yield (self.__training_features__[start: end], self.__training_targets__[start: end])

    def get_cross_validation_features_and_targets(self):

        """

        :return:
        """
        cv_sequences_length = self.__cv_features__.shape[0]
        if self.__batch__ > cv_sequences_length:
            raise Exception('Batch size is too big, consider low down batch size')
        while True:
            if self.__random__:
                index = np.random.randint(
                    low=0,
                    high=cv_sequences_length,
                    size=self.__batch__)
                yield (self.__cv_features__[index], self.__cv_targets__[index])
            else:
                start = self.__cv_pointer__ * self.__batch__
                end = (self.__cv_pointer__ + 1) * self.__batch__
                # Pointer greater than the total length of the training data
                # 如果一旦开始或者终止的index越界，那么就归零，并且返回最上面的那一串数据
                # 这样的坏处是，最后那一小部分除不净的数据永远也用不到了，就好像被删除了
                if start > cv_sequences_length or end > cv_sequences_length:
                    self.__cv_pointer__ = 0
                    yield (self.__cv_features__[0: self.__batch__], self.__cv_targets__[0: self.__batch__])
                else:
                    yield (self.__cv_features__[start: end], self.__cv_targets__[start: end])

    def get_test_features_and_targets(self):
        """

        :return:
        """
        test_sequences_length = self.__test_features__.shape[0]
        if self.__batch__ > test_sequences_length:
            raise Exception('Batch size is too big, consider low down batch size')
        while True:
            if self.__random__:
                index = np.random.randint(
                    low=0,
                    high=test_sequences_length,
                    size=self.__batch__)
                yield (self.__test_features__[index], self.__test_targets__[index])
            else:
                start = self.__test_pointer__ * self.__batch__
                end = (self.__test_pointer__ + 1) * self.__batch__
                # Pointer greater than the total length of the training data
                # 如果一旦开始或者终止的index越界，那么就归零，并且返回最上面的那一串数据
                # 这样的坏处是，最后那一小部分除不净的数据永远也用不到了，就好像被删除了
                if start > test_sequences_length or end > test_sequences_length:
                    self.__test_pointer__ = 0
                    yield (self.__test_features__[0: self.__batch__], self.__test_targets__[0: self.__batch__])
                else:
                    yield (self.__test_features__[start: end], self.__test_targets__[start: end])
