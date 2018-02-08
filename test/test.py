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

for i in range(10):
    print('tf :', container.get_training_features()[0][0])
    print('tt :', container.get_training_targets()[0][0])
    print('cf :', container.get_cv_features()[0][0])
    print('ct :', container.get_cv_targets()[0][0])
    print('gf :', container.get_test_features()[0][0])
    print('gt :', container.get_test_targets()[0][0])

df_1 = pd.DataFrame(np.reshape(np.arange(300), (3, -1)).T, columns=['a', 'b', 'c'])

container_classification = RNNContainer(df_1)

container_classification.set_feature_tags(['a', 'b'])
container_classification.set_target_tags(['c'])
print('++++++++++++++++++++++')
print('shape: ', container_classification.data.shape)
container_classification.gen_batch_for_sequence_classification(batch=2, time_steps=5, randomly=False)
print('++++++++++++++++++++++')
for _ in range(1000):
    print('tf :', container_classification.get_training_features())
    print('tt :', container_classification.get_training_targets())
    print('cf :', container_classification.get_cv_features())
    print('ct :', container_classification.get_cv_targets())
    # print('gf :', container_classification.get_test_features()[0][0])
    # print('gt :', container_classification.get_test_targets()[0][0])
