import unittest
import numpy as np
import pandas as pd
from data_summarizer import DataSummarizer


class TestSummarizer(unittest.TestCase):
    def test_empty_df(self):
        summarizer = DataSummarizer()
        summary = summarizer.summarize(pd.DataFrame())
        self.assertTrue(summary.empty)

    def test_num_stats(self):
        df = pd.DataFrame({'col1': [1, 3, 2, 2],
                           'col2': [0, -4, 3, 4]})
        summarizer = DataSummarizer()
        summary = summarizer.summarize(df)

        self.assertFalse(summarizer.summary.empty)
        self.assertTrue(np.array_equal(summary.loc['min'].values,
                                       np.array([1.0, -4.0])))
        self.assertTrue(np.array_equal(summary.loc['max'].values,
                                       np.array([3.0, 4.0])))
        self.assertTrue(np.array_equal(summary.loc['mean'].values,
                                       np.array([2.0, 0.75])))
        self.assertTrue(np.array_equal(summary.loc['median'].values,
                                       np.array([2.0, 1.5])))
        self.assertTrue(np.array_equal(summary.loc['mode'].values,
                                       np.array([2.0, -4.0])))
        self.assertTrue(np.allclose(summary.loc['var'].values.astype('float32'),
                                    np.array([0.666667, 12.916667])))
        self.assertTrue(np.allclose(summary.loc['std'].values.astype('float32'),
                                    np.array([0.816497, 3.593976])))
        self.assertTrue(np.array_equal(summary.loc['Q1'].values,
                                       np.array([1.75, -1.0])))
        self.assertTrue(np.array_equal(summary.loc['Q3'].values,
                                       np.array([2.25, 3.25])))
        self.assertTrue(np.array_equal(summary.loc['IQR'].values,
                                       np.array([0.5, 4.25])))
        self.assertTrue(np.array_equal(summary.loc['% NaN'].values,
                                       np.array([0.0, 0.0])))
        self.assertTrue(np.array_equal(summary.loc['unique'].values,
                                       np.array([3.0, 4.0])))
        self.assertTrue(np.array_equal(summary.loc['count'].values,
                                       np.array([4.0, 4.0])))

    def test_obj_stats(self):
        df = pd.DataFrame({'col1': ['aa', 'bb', '22', 5],
                           'col2': ['aa', 'bb', 'aa', 'dd']})

        summarizer = DataSummarizer()
        summary = summarizer.summarize(df)
        self.assertFalse(summarizer.summary.empty)
        self.assertTrue(np.array_equal(summary.loc['top'].values,
                                       np.array(['aa', 'aa'])))
        self.assertTrue(np.array_equal(summary.loc['freq'].values,
                                       np.array([0.25, 0.5])))
        self.assertTrue(np.array_equal(summary.loc['unique'].values,
                                       np.array([4.0, 3.0])))
        self.assertTrue(np.array_equal(summary.loc['count'].values,
                                       np.array([4.0, 4.0])))

    def test_cat_stats(self):
        df = pd.DataFrame({'col1': ['aa', 'bb', 'cc', 'cc'],
                           'col2': ['aa', 'bb', 'aa', 'dd']},
                          dtype='category')

        summarizer = DataSummarizer()
        summary = summarizer.summarize(df)
        self.assertFalse(summarizer.summary.empty)
        self.assertTrue(np.array_equal(summary.loc['top'].values,
                                       np.array(['cc', 'aa'])))
        self.assertTrue(np.array_equal(summary.loc['freq'].values,
                                       np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(summary.loc['unique'].values,
                                       np.array([3.0, 3.0])))
        self.assertTrue(np.array_equal(summary.loc['count'].values,
                                       np.array([4.0, 4.0])))

    def test_time_stats(self):
        df = pd.DataFrame({'col1': [pd.Timestamp('2018-01-05'),
                                    pd.Timestamp('2018-01-05'),
                                    pd.Timestamp('2018-05-05')],
                           'col2': [pd.Timedelta('1 days 2 hours'),
                                    pd.Timedelta('1 days 2 hours'),
                                    pd.Timedelta('1 days 5 hours')]})

        summarizer = DataSummarizer()
        summary = summarizer.summarize(df)
        self.assertFalse(summarizer.summary.empty)
        self.assertTrue(np.array_equal(summary.loc['min'].values,
                                       np.array([pd.Timestamp('2018-01-05'),
                                                 pd.Timedelta('1 days 2 hours')])))
        self.assertTrue(np.array_equal(summary.loc['max'].values,
                                       np.array([pd.Timestamp('2018-05-05'),
                                                 pd.Timedelta('1 days 5 hours')])))
        self.assertTrue(np.array_equal(summary.loc['mean'].values,
                                       np.array([pd.Timestamp('2018-02-14'),
                                                 pd.Timedelta('1 days 3 hours')])))
        self.assertTrue(np.array_equal(summary.loc['median'].values,
                                       np.array([pd.Timestamp('2018-01-05'),
                                                 pd.Timedelta('1 days 2 hours')])))
        self.assertTrue(np.array_equal(summary.loc['mode'].values,
                                       np.array([pd.Timestamp('2018-01-05'),
                                                 pd.Timedelta('1 days 2 hours')])))
        self.assertTrue(np.allclose(summary.loc['var'].values.astype('float32'),
                                    np.array([np.NaN, np.NaN]), equal_nan=True))
        self.assertTrue(np.allclose(summary.loc['std'].values.astype('float32'),
                                    np.array([np.NaN, np.NaN]), equal_nan=True))
        self.assertTrue(np.array_equal(summary.loc['Q1'].values,
                                       np.array([pd.Timestamp('2018-01-05'),
                                                 pd.Timedelta('1 days 2 hours')])))
        self.assertTrue(np.array_equal(summary.loc['Q3'].values,
                                       np.array([pd.Timestamp('2018-03-06'),
                                                 pd.Timedelta('1 days 3 hours 30 minutes')])))
        self.assertTrue(np.array_equal(summary.loc['IQR'].values,
                                       np.array([pd.Timedelta('60 days'),
                                                 pd.Timedelta('1 hour 30 minutes')])))
        self.assertTrue(np.array_equal(summary.loc['% NaN'].values,
                                       np.array([0.0, 0.0])))
        self.assertTrue(np.array_equal(summary.loc['unique'].values,
                                       np.array([2.0, 2.0])))
        self.assertTrue(np.array_equal(summary.loc['count'].values,
                                       np.array([3.0, 3.0])))

    def test_comb_stats(self):
        df = pd.DataFrame({'col1': [1, 2, 3],
                           'col2': [pd.Timestamp('2018-01-05'),
                                    pd.Timestamp('2018-02-05'),
                                    pd.Timestamp('2018-03-05')],
                           'col3': ['a', 'b', 'c']})
        summarizer = DataSummarizer()
        summary = summarizer.summarize(df)
        self.assertFalse(summarizer.summary.empty)
        self.assertTrue(np.array_equal(summary.isna().sum().values,
                                       np.array([2.0, 4.0, 11.0])))
