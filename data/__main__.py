import random

import numpy as np
from sklearn import preprocessing


class Data:
    def __init__(self,
                 data,
                 training_set_split_ratio=0.7,
                 cross_validation_set_split_ratio=0.2,
                 test_set_split_ratio=0.1):
        self.data = data
        self.__data__ = data
        self.scale = preprocessing.StandardScaler()
        self.normalizer = self.scale.fit(data)
        self.feature_tags = []
        self.target_tags = []

        # take control over randomness
        self.__random_seed__ = random.seed(a=10)

        # create train/cv/test dataSet
        self.training_set_split_ratio = training_set_split_ratio
        self.cross_validation_set_split_ratio = cross_validation_set_split_ratio
        self.test_set_split_ratio = test_set_split_ratio

        # define pseudo mask for generating class
        mask = np.random.rand(len(data))
        self.training_set_mask = mask <= self.training_set_split_ratio
        self.cross_validation_set_mask = (mask > self.training_set_split_ratio) & \
                                         (
                                         mask < (self.training_set_split_ratio + self.cross_validation_set_split_ratio))
        self.test_set_mask = mask >= (self.training_set_split_ratio + self.cross_validation_set_split_ratio)

    def detect_null(self):
        return self.data.isnull().any()

    def restore(self):
        self.data = self.__data__
        return self

    def interpolate(self, data, method=None):
        self.__data__ = self.data
        self.data = data.interpolate(method=method)
        return self

    def set_feature_tags(self, feature_tags):
        self.feature_tags = feature_tags
        return self

    def set_target_tags(self, target_tags):
        self.target_tags = target_tags
        return self

    def add_feature_tag(self, tag):
        self.feature_tags.append(tag)
        return self

    def add_target_tag(self, tag):
        self.target_tags.append(tag)
        return self

    def fit(self, data):
        self.normalizer = self.scale.fit(data)
        return self

    def normalize(self, data):
        return self.normalizer.transform(data)

    def get_feature_data(self):
        pass
        return self

    def get_target_data(self):
        pass
        return self

    def get_training_features(self):
        pass
        return self

    def get_training_targets(self):
        pass
        return self

    def get_cross_validation_features(self):
        pass
        return self

    def get_cross_validation_targets(self):
        pass
        return self

    def get_test_features(self):
        pass
        return self

    def get_test_targets(self):
        pass
        return self
