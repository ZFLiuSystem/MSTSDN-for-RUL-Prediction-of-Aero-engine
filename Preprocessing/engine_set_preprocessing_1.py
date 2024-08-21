import numpy as np
import pandas as pd
from sklearn import preprocessing


class CmaPss:
    def __init__(self, file_name: str, rul_early: int, sequence_length: int, test_sequence: int, graph_signals: bool):
        self.graph_signals = graph_signals
        self.train_df = pd.read_csv('file address' + file_name + '.txt', sep=' ', header=None)
        self.train_df.drop(columns=[26, 27], axis=1, inplace=True)
        self.train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6',
                                 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
                                 's20', 's21']
        self.train_df.sort_values(['id', 'cycle'], inplace=True)
        self.test_df = pd.read_csv('file address' + file_name + '.txt', sep=" ", header=None)
        self.test_df.drop(self.test_df.columns[[26, 27]], axis=1, inplace=True)
        self.test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6',
                                's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
                                's20', 's21']
        self.truth_df = pd.read_csv('file address' + file_name + '.txt', sep=" ", header=None)
        self.truth_df.drop(self.truth_df.columns[1], axis=1, inplace=True)
        self.R_early = rul_early
        self.sequence_length = sequence_length
        self.test_sequence = test_sequence

    @staticmethod
    def column_sequence_selection():
        sequence_col = ['cycle_norm', 'setting1',
                        'setting2', 'setting3'] + ['s'+str(i) for i in range(1, 22)]
        sequence_col = [x for x in sequence_col if x not in ['cycle_norm', 'id', 'setting1', 'setting2', 'setting3',
                                                             's1', 's5', 's6', 's10', 's16', 's18', 's19']]
        return sequence_col

    def train_df_prep(self):
        rul = pd.DataFrame(self.train_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.train_df = self.train_df.merge(rul, on=['id'], how='left')
        self.train_df['rul'] = self.train_df['max'] - self.train_df['cycle']
        self.train_df.drop('max', axis=1, inplace=True)
        self.train_df['RUL_early'] = self.train_df['rul'].apply(lambda x: self.R_early if x >= self.R_early
                                                                else x)
        train_df = self.train_df.drop('rul', axis=1)
        return train_df

    def test_df_prep(self):
        rul = pd.DataFrame(self.test_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.truth_df.columns = ['more']
        self.truth_df['id'] = self.truth_df.index + 1
        self.truth_df['max'] = rul['max'] + self.truth_df['more']
        self.truth_df.drop('more', axis=1, inplace=True)
        self.test_df = self.test_df.merge(self.truth_df, on=['id'], how='left')
        self.test_df['rul'] = self.test_df['max']-self.test_df['cycle']
        self.test_df.drop('max', axis=1, inplace=True)
        self.test_df['RUL_early'] = self.test_df['rul'].apply(lambda x: self.R_early if x >= self.R_early else x)
        test_df = self.test_df.drop('rul', axis=1)
        return test_df

    def selection_and_normal(self, train_df, test_df):
        min_max_normal = preprocessing.MinMaxScaler()
        train_df['cycle_norm'] = train_df['cycle']
        normal_col = train_df.columns.difference(['id', 'cycle', 'RUL_early'])
        normal_col_train_df = pd.DataFrame(min_max_normal.fit_transform(train_df[normal_col]), columns=normal_col,
                                           index=train_df.index)
        join_df = train_df[train_df.columns.difference(normal_col)].join(normal_col_train_df)
        train_df = join_df.reindex(columns=train_df.columns)

        test_df['cycle_norm'] = test_df['cycle']
        normal_col = test_df.columns.difference(['id', 'cycle', 'RUL_early'])
        nor_col_test_df = pd.DataFrame(min_max_normal.transform(test_df[normal_col]), columns=normal_col,
                                       index=test_df.index)
        join_df = test_df[test_df.columns.difference(normal_col)].join(nor_col_test_df)
        test_df = join_df.reindex(columns=test_df.columns)
        test_df = test_df.reset_index(drop=True)

        seq = self.column_sequence_selection()
        static_features = np.asarray(train_df[seq], dtype=np.float32)
        return train_df, test_df, static_features

    def gen_train_samples_prep(self, data_df):
        seq_col = self.column_sequence_selection()
        data_matrix = data_df[seq_col].values

        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements-self.sequence_length), range(self.sequence_length, num_elements)):
            yield data_matrix[start:stop, :]

    def gen_train_samples(self, data_df):
        train_samples = (list(self.gen_train_samples_prep(data_df[data_df['id'] == id_i]))
                         for id_i in data_df['id'].unique())
        train_samples = np.concatenate(list(train_samples)).astype(np.float32)
        return train_samples

    def gen_label_prep(self, data_df):
        data_matrix = data_df[['RUL_early']].values
        num_elements = data_matrix.shape[0]
        return data_matrix[self.sequence_length:num_elements, :]

    def gen_labels(self, data_df):
        train_labels = (list(self.gen_label_prep(data_df[data_df['id'] == id_i]))
                        for id_i in data_df['id'].unique())
        train_labels = np.concatenate(list(train_labels)).astype(np.float32)
        return train_labels

    def gen_samples_labels(self):
        train_df = self.train_df_prep()
        test_df = self.test_df_prep()
        train_df, test_df, static_features = self.selection_and_normal(train_df, test_df)
        train_samples = self.gen_train_samples(train_df)
        train_labels = self.gen_labels(train_df)
        sequence_col = self.column_sequence_selection()
        seq_array_test_last = [
            test_df[test_df["id"] == id_i][sequence_col].values[-self.test_sequence:]
            for id_i in test_df["id"].unique()
            if len(test_df[test_df["id"] == id_i]) >= self.test_sequence
        ]
        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

        y_mask = [
            len(test_df[test_df["id"] == id_i]) >= self.test_sequence
            for id_i in test_df["id"].unique()
        ]
        label_array_test_last = test_df.groupby("id")["RUL_early"].nth(-1)[y_mask].values
        label_array_test_last = label_array_test_last.reshape(
            label_array_test_last.shape[0], 1
        ).astype(np.float32)
        if self.graph_signals:
            return train_samples, train_labels, seq_array_test_last, label_array_test_last, static_features
        else:
            return train_samples, train_labels, seq_array_test_last, label_array_test_last
