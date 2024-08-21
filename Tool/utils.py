import torch
import numpy as np
from engine_set_preprocessing_1 import CmaPss
from engine_set_preprocessing_2 import NCmaPss


class Dataloader(object):
    def __init__(self, train_samples, train_labels, batch_size: int, train_mode: bool):
        self.train_x = train_samples
        self.train_y = train_labels
        self.batch_size = batch_size
        self.len_train_x = len(train_samples)
        self.num_batch = int(self.len_train_x // batch_size)
        self.train_mode = train_mode
        self.current_batch_ind = 0

    def shuffle_train_set(self):
        permutation = np.random.permutation(self.len_train_x)
        train_x, train_y = self.train_x[permutation], self.train_y[permutation]
        self.train_x, self.train_y = train_x, train_y

    def get_iterator(self):
        self.current_batch_ind = 0
        if self.train_mode:
            self.shuffle_train_set()

        def _wrapper():
            while True:
                if self.current_batch_ind > self.num_batch:
                    break
                start_ind = self.batch_size * self.current_batch_ind
                end_ind = min(self.len_train_x, (self.current_batch_ind + 1) * self.batch_size)

                train_x_i = self.train_x[start_ind:end_ind]
                train_y_i = self.train_y[start_ind:end_ind]
                yield train_x_i, train_y_i
                self.current_batch_ind += 1

        return _wrapper()


def load_dataset(dataset_name: str, rul_early: int, sequence_length: int, test_sequence: int,
                 batch_size: int, valid_batch_size=None, valid_step=None, test_batch_size=None):
    data = {}
    ratio = None

    if 'FD' in dataset_name:
        get_data = CmaPss(file_name=dataset_name, sequence_length=sequence_length, test_sequence=test_sequence,
                          rul_early=rul_early)
    elif 'DS' in dataset_name:
        get_data = NCmaPss(file_name=dataset_name, sequence_length=sequence_length, test_sequence=test_sequence,
                           rul_early=rul_early)
    else:
        raise ValueError('The name of data file is not correct !')

    samples, labels, test_samples, test_labels = get_data.gen_samples_labels()
    print(f"Training samples size: {samples.shape}.")
    print(f"Training labels size: {labels.shape}.")
    length_data = len(samples)
    train_samples, train_labels = samples[:int(ratio * length_data)], labels[:int(ratio * length_data)]
    val_samples, val_labels = samples[int(ratio * length_data):], labels[int(ratio * length_data):]
    data['train_samples'], data['train_labels'] = train_samples, train_labels
    data['val_samples'], data['val_labels'] = val_samples, val_labels
    data['test_samples'], data['test_labels'] = test_samples, test_labels
    data['train_loader'] = Dataloader(data['train_samples'], data['train_labels'], batch_size=batch_size)
    data['val_loader'] = Dataloader(data['val_samples'], data['val_labels'],
                                    batch_size=valid_batch_size, train_mode=False)
    return data
    pass


def huberloss_normal(predictions: list, labels: list, delta: float, reduction: str):
    loss_list = []
    for prediction, label in zip(predictions, labels):
        difference = torch.abs(prediction - label)
        if difference.item() < delta:
            loss_i = 0.5 * torch.square(prediction - label)
        else:
            loss_i = delta * (difference - 0.5 * delta)
        loss_list.append(loss_i)
    loss_list = torch.tensor(loss_list, dtype=torch.float, requires_grad=True)
    if reduction == 'mean':
        return torch.mean(loss_list)
    elif reduction == 'sum':
        return torch.sum(loss_list)
    else:
        return loss_list
    pass


def get_mse(predictions, labels):
    loss = torch.square((predictions - labels))
    return torch.mean(loss)
    pass


def get_root_mse(predictions, labels):
    mse = get_mse(predictions, labels)
    return torch.sqrt(mse)
    pass


def get_score(predictions, labels):
    score_list = []
    for prediction, label in zip(predictions, labels):
        difference = prediction - label
        if difference.item() < 0:
            difference /= (-10)
            score_list.append(torch.exp(difference) - 1.0)
        else:
            difference /= 13
            score_list.append(torch.exp(difference) - 1.0)
    score_list = torch.tensor(score_list, dtype=torch.float)
    return torch.sum(score_list)
    pass


def get_metrics(predictions, labels):
    root_mse = get_root_mse(predictions, labels).item()
    score = get_score(predictions, labels).item()
    return root_mse, score
    pass

