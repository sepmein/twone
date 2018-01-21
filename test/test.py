import numpy as np
import pandas as pd

from twone import RNNContainer

df = pd.DataFrame(np.reshape(np.arange(30003), (3, -1)).T, columns=['a', 'b', 'c'])

feature_tags = ['a', 'b']
target_tags = ['c']

container = RNNContainer(df)

container.set_feature_tags(feature_tags)
container.set_target_tags(target_tags)

container.gen_batch_for_sequence_labeling(batch=5, time_steps=100)

for i in range(100):
    print('tf :', container.get_training_features()[0][0])
    print('tt :', container.get_training_targets()[0][0])
    print('cf :', container.get_cv_features()[0][0])
    print('ct :', container.get_cv_targets()[0][0])
    print('gf :', container.get_test_features()[0][0])
    print('gt :', container.get_test_targets()[0][0])
