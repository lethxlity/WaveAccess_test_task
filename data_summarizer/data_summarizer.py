import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


class DataSummarizer:
    pandas_dtypes = {'num': ['int8', 'int16', 'int32', 'int64', 'float16', 'float32',
                             'float64', 'datetime64[ns]', 'timedelta64[ns]'],
                     'cat': ['object', 'bool', 'category']}

    def __init__(self):
        self.df = pd.DataFrame()
        self.num_columns = []
        self.cat_columns = []
        self.summary = pd.DataFrame()

    def check_dtypes(self):
        for col in self.df:
            col_dtype = self.df[col].dtype
            if col_dtype in self.pandas_dtypes['num']:
                self.num_columns.append(col)
            elif col_dtype in self.pandas_dtypes['cat']:
                self.cat_columns.append(col)
            self.summary.at[col, 'dtype'] = self.df[col].dtype

    def calc_numerical_stats(self):
        for col in self.num_columns:
            self.summary.at[col, 'min'] = self.df[col].min()
            self.summary.at[col, 'max'] = self.df[col].max()
            self.summary.at[col, 'mean'] = self.df[col].mean()
            self.summary.at[col, 'median'] = self.df[col].median()
            self.summary.at[col, 'mode'] = self.df[col].mode()[0]

            if not (np.issubdtype(self.df[col].dtype, np.datetime64) or
                    np.issubdtype(self.df[col].dtype, np.timedelta64)):
                self.summary.at[col, 'var'] = self.df[col].var()
                self.summary.at[col, 'std'] = self.df[col].std()
            else:
                self.summary.at[col, 'var'] = np.NaN
                self.summary.at[col, 'std'] = np.NaN

            self.summary.at[col, 'Q1'] = self.df[col].quantile(0.25)
            self.summary.at[col, 'Q3'] = self.df[col].quantile(0.75)
            self.summary.at[col, 'IQR'] = self.summary.at[col, 'Q3'] - self.summary.at[col, 'Q1']

            self.summary.at[col, '% NaN'] = self.df[col].isna().sum()
            self.summary.at[col, 'unique'] = self.df[col].nunique()
            self.summary.at[col, 'count'] = self.df[col].count()

    def calc_categorical_stats(self):
        for col in self.cat_columns:
            value_counts = self.df[col].value_counts()
            self.summary.at[col, 'top'] = value_counts.index[0]
            self.summary.at[col, 'freq'] = value_counts.values[0] / value_counts.sum()

            self.summary.at[col, 'unique'] = self.df[col].nunique()
            self.summary.at[col, 'count'] = self.df[col].count()

    def summarize(self, df):
        self.df = df.copy()
        self.check_dtypes()
        self.calc_numerical_stats()
        self.calc_categorical_stats()
        return self.summary.transpose()

    def save_to_disk(self, output_type='html', output_name='summary', decimals=2):
        summary_to_save = self.summary.round(2).transpose()
        filepath = output_name + '.' + output_type
        if output_type == 'xlsx':
            summary_to_save.to_excel(filepath, index=True)
        elif output_type == 'html':
            with open(filepath, "w") as report:
                report.write(summary_to_save.to_html())
        elif output_type == 'md':
            with open(filepath, "w") as report:
                report.write(summary_to_save.to_markdown())


if __name__ == '__main__':

    df = load_iris(as_frame=True).frame
    df['target'] = df['target'].astype('category')

    summarizer = DataSummarizer()
    summarizer.summarize(df)
    summarizer.save_to_disk(output_type='md')
