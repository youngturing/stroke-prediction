from unittest import TestCase

import pandas as pd


class TestData(TestCase):
    data_path = '../data/stroke_data_transformed.csv'
    data_types = [
        'float64',
        'int64',
        'int64',
        'int64',
        'float64',
        'int64',
    ]
    column_names = [
        'age',
        'hypertension',
        'heart_disease',
        'ever_married',
        'avg_glucose_level',
        'stroke',
    ]

    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(cls.data_path)

    def test_data_types(self):
        for dtypes, dtypes_loaded in zip(self.data_types, self.data.dtypes):
            self.assertEqual(dtypes, dtypes_loaded)

    def test_colum_names(self):
        for col_names, loaded_col_names in zip(self.column_names, self.data.columns):
            self.assertEqual(col_names, loaded_col_names)
