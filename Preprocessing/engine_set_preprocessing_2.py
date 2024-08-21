import numpy as np
import pandas as pd
from sklearn import preprocessing


class NCmaPss:
    def __init__(self, file_name: str, rul_early: int, sequence_length: int, test_sequence: int, graph_signals: bool):
        self.graph_signals = graph_signals
        self.train_df = pd.read_csv("file address" + file_name + '_set_unit_train.csv', header=None)
        self.train_df.columns = ['id', 'flight_cycle'] + [f's{i}' for i in range(1, 29)] + ['RUL']
        self.test_df = pd.read_csv("file address" + file_name + '_set_unit_test.csv', header=None)
        self.test_df.columns = ['id', 'flight_cycle'] + [f's{i}' for i in range(1, 29)] + ['RUL']
        self.R_early = rul_early
        self.sequence_length = sequence_length
        self.test_sequence = test_sequence
    
    def train_df_prep(self):
        '''
        rul = pd.DataFrame(self.train_df.groupby('flight_cycle')['flight_cycle_id'].max()).reset_index()
        rul.columns = ['flight_cycle', 'max']
        self.train_df = self.train_df.merge(rul, on=['flight_cycle'], how='left')
        self.train_df['rul'] = self.train_df['max'] - self.train_df['flight_cycle_id']
        self.train_df.drop('max', axis=1, inplace=True)
        '''
        self.train_df['RUL_label'] = self.train_df['RUL'].apply(lambda x: self.R_early if x >= self.R_early
                                                                else x)
        train_df = self.train_df.drop('RUL', axis=1)
        return train_df

    def test_df_prep(self):
        '''
        rul = pd.DataFrame(self.test_df.groupby('flight_cycle')['flight_cycle_id'].max()).reset_index()
        rul.columns = ['flight_cycle', 'max']
        self.truth_df.columns = ['more']
        self.truth_df['flight_cycle'] = self.truth_df.index + 1
        self.truth_df['max'] = rul['max'] + self.truth_df['more']
        self.truth_df.drop('more', axis=1, inplace=True)
        self.test_df = self.test_df.merge(self.truth_df, on=['flight_cycle'], how='left')
        self.test_df['rul'] = self.test_df['max']-self.test_df['flight_cycle_id']
        self.test_df.drop('max', axis=1, inplace=True)
        '''
        self.test_df['RUL_label'] = self.test_df['RUL'].apply(lambda x: self.R_early if x >= self.R_early 
                                                              else x)
        test_df = self.test_df.drop('RUL', axis=1)
        return test_df  
    
    def selection_and_normal(self, train_df, test_df):
        min_max_normal = preprocessing.MinMaxScaler()
        normal_col = train_df.columns.difference(['id', 'flight_cycle', 'RUL_label'])
        normal_col_train_df = pd.DataFrame(min_max_normal.fit_transform(train_df[normal_col]), columns=normal_col,
                                           index=train_df.index)
        join_df = train_df[train_df.columns.difference(normal_col)].join(normal_col_train_df)
        train_df = join_df.reindex(columns=train_df.columns)

        normal_col = test_df.columns.difference(['id', 'flight_cycle', 'RUL_label'])
        nor_col_test_df = pd.DataFrame(min_max_normal.transform(test_df[normal_col]), columns=normal_col,
                                       index=test_df.index)
        join_df = test_df[test_df.columns.difference(normal_col)].join(nor_col_test_df)
        test_df = join_df.reindex(columns=test_df.columns)
        test_df = test_df.reset_index(drop=True)
        return train_df, test_df

    def gen_train_samples_prep(self, data_df):
        seq_col = data_df.columns.difference(['id', 'flight_cycle', 'RUL_label'])
        data_matrix = data_df[seq_col].values
        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements-self.sequence_length), range(self.sequence_length, num_elements)):
            yield data_matrix[start:stop, :]

    def gen_train_samples(self, data_df):
        train_samples = (list(self.gen_train_samples_prep(data_df[data_df['flight_cycle'] == id_i]))
                         for id_i in data_df['flight_cycle'].unique())
        train_samples = np.concatenate(list(train_samples)).astype(np.float32)
        return train_samples

    def gen_label_prep(self, data_df):
        data_matrix = data_df[['RUL_label']].values
        num_elements = data_matrix.shape[0]
        return data_matrix[self.sequence_length:num_elements, :]

    def gen_labels(self, data_df):
        train_labels = (list(self.gen_label_prep(data_df[data_df['flight_cycle'] == id_i]))
                        for id_i in data_df['flight_cycle'].unique())
        train_labels = np.concatenate(list(train_labels)).astype(np.float32)
        return train_labels

    def get_seq_last_array(self, test_df):
        sequence_col = test_df.columns.difference(['id', 'flight_cycle', 'RUL_label'])
        seq_array_test_last = [
            test_df[test_df['flight_cycle'] == id_i][sequence_col].values[-self.test_sequence:]
            for id_i in test_df['flight_cycle'].unique()
            if len(test_df[test_df['flight_cycle'] == id_i]) >= self.test_sequence
        ]
        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
        return seq_array_test_last

    def get_label_last_array(self, test_df):
        y_mask = [
            len(test_df[test_df['flight_cycle'] == id_i]) >= self.test_sequence
            for id_i in test_df['flight_cycle'].unique()
        ]
        label_array_test_last = test_df.groupby('flight_cycle')["RUL_label"].nth(-1)[y_mask].values
        label_array_test_last = label_array_test_last.reshape(
            label_array_test_last.shape[0], 1
        ).astype(np.float32)
        return label_array_test_last

    def gen_samples_labels(self):
        train_df = self.train_df_prep()
        test_df = self.test_df_prep()
        train_df, test_df = self.selection_and_normal(train_df, test_df)
        i = 1
        list_id = train_df['id'].unique()
        start_train_df = train_df[train_df['id'] == list_id[0]]
        start_train_df.columns = train_df.columns
        start_train_samples = self.gen_train_samples(start_train_df)
        start_train_labels = self.gen_labels(start_train_df)
        while True:
            i_train_df = train_df[train_df['id'] == list_id[i]]
            i_train_df.columns = train_df.columns
            i_train_samples = self.gen_train_samples(i_train_df)
            i_train_labels = self.gen_labels(i_train_df)
            start_train_samples = np.concatenate([start_train_samples, i_train_samples], axis=0)
            start_train_labels = np.concatenate([start_train_labels, i_train_labels], axis=0)
            i += 1
            if i > len(list_id) - 1:
                break
        j = 1
        list_id = test_df['id'].unique()
        start_test_df = test_df[test_df['id'] == list_id[0]]
        start_test_df.columns = test_df.columns
        start_test_samples = self.get_seq_last_array(start_test_df)
        start_test_labels = self.get_label_last_array(start_test_df)
        while True:
            j_test_df = test_df[test_df['id'] == list_id[j]]
            j_test_df.columns = test_df.columns
            j_test_samples = self.get_seq_last_array(j_test_df)
            j_test_labels = self.get_label_last_array(j_test_df)
            start_test_samples = np.concatenate([start_test_samples, j_test_samples], axis=0)
            start_test_labels = np.concatenate([start_test_labels, j_test_labels], axis=0)
            j += 1
            if j > len(list_id) - 1:
                break
        if self.graph_signals:
            return start_train_samples, start_train_labels, start_test_samples, start_test_labels
        else:
            return start_train_samples, start_train_labels, start_test_samples, start_test_labels
            