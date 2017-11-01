import numpy as np
import pandas as pd
from sklearn import preprocessing


class Preprocessor:
    def __init__(self, data):
        self.scale = preprocessing.StandardScaler()
        if data is not None:
            self.normalizer = self.scale.fit(data)
        else:
            self.normalizer = None

    def detect_null(self):
        pass

    def interpolate(self, method):
        pass

    def fit(self, data):
        self.normalizer = self.scale.fit(data)
        pass

    def normalize(self, input):
        pass

