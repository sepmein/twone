import numpy as np
import pandas as pd

from twone import RNNContainer

df = pd.DataFrame(np.reshape(np.arange(30003), (3, -1)).T, columns=['a', 'b', 'c'])

feature_tags = ['a', 'b']
target_tags = ['c']

container = RNNContainer(df)

container.set_feature_tags(feature_tags)
container.set_target_tags(target_tags)

container.gen_batch(batch=5, time_steps=100)

for i in range(100):
    print(container.get_training_features()[0][0])
    print(container.get_training_targets()[0][0])
