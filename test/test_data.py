import unittest

import pandas as pd

from twone.container import Container, RNNContainer


class ClassDataTestCase(unittest.TestCase):
    """Test for data/data.py"""

    def test_instantiate(self):
        weather_death_data = pd.read_csv('../interpolated_data.csv')
        data = Container(data_frame=weather_death_data)
        self.assertIsInstance(data, Container)

    def test_test(self):
        self.assertTrue(True)

    def test_load_data(self):
        feature_tags = ["temp", "rh", "so2", "no2", "co", "pm10", "pm2.5", "o3", "day", "dow", "death_total"]
        target_tags = ['death_total']
        data = pd.read_csv('../interpolated_data.csv')
        container = RNNContainer(data_frame=data)
        self.assertIsInstance(container, RNNContainer)
        container.set_feature_tags(
            feature_tags) \
            .set_target_tags(target_tags) \
            .compute_feature_data() \
            .compute_target_data()

        self.assertEqual(container.__feature_data__.shape[1], len(feature_tags))
        self.assertEqual(container.__target_data__.shape[1], len(target_tags))

        feature_data = container.get_feature_data(5)
        print(feature_data)
        print(feature_data.shape)


if __name__ == '__main__':
    unittest.main()
