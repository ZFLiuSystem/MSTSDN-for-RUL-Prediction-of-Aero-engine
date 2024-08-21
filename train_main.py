import torch
import numpy as np
import argparse
import utils
import time
from tqdm import tqdm
from engine_train import Trainer


def arg_option():
    parser = argparse.ArgumentParser("The required parameters of the model")
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--data', default=None, type=str)
    parser.add_argument('--graphs_signals', default=None, type=bool)
    parser.add_argument('--num_nodes', default=None, type=int)
    parser.add_argument('--num_features', default=None, type=int)
    parser.add_argument('--test_sequence', default=None, type=int)
    parser.add_argument('--dim_k', default=int(None), type=int)
    parser.add_argument('--t_heads', default=None, type=list)
    parser.add_argument('--window_size', default=None, type=list)
    parser.add_argument('--mlp_ratio', default=None, type=float)
    parser.add_argument('--adj_dim', default=None, type=int)
    parser.add_argument('--r_mlp_ratio', default=None, type=float)
    parser.add_argument('--r_mlp_dropout', default=None, type=float)
    parser.add_argument('--RUL_early', default=None, type=int)
    parser.add_argument('--drop_rate', default=None, type=float)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--milestones', default=None, type=list)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--patience_epochs', default=None, type=int)
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--save_best', default=None, type=str)
    parser.add_argument('--exp_id', default=None, type=int)
    args = parser.parse_args()
    return args


def main():
    args = arg_option()
    device = torch.device(args.device)
    data = utils.load_dataset(dataset_name=args.data, rul_early=args.RUL_early,
                              sequence_length=args.num_features, test_sequence=args.test_sequence,
                              batch_size=args.batch_size, valid_batch_size=args.batch_size)

    print(args)

    training = Trainer(device=device, lr_rate=args.learning_rate,
                       weight_decay=args.weight_decay, nodes=args.num_nodes,
                       seq=args.num_features, dim_k=args.dim_k,
                       heads=args.t_heads, window_size=args.window_size,
                       mlp_ratio=args.mlp_ratio, adj_dim=args.adj_dim,
                       r_mlp_ratio=args.r_mlp_ratio, dropout_rate=args.drop_rate,
                       milestones=args.milestones, r_mlp_dropout=args.r_mlp_dropout)

    lr_schedule = training.lr_schedule
    train_time = []
    valid_time = []
    train_loss = []
    valid_loss = []

    for i in range(1, args.epochs + 1):
        train_batches_loss = []
        train_root_mse = []
        train_score = []
        t_start = time.time()
        data['train_loader'].shuffle_train_set()
        print(90 * '-')
        for no_batch, (samples_tra, labels) in enumerate(tqdm((data['train_loader'].get_iterator()))):
            if len(samples_tra) != args.batch_size:
                break
            samples_tra = torch.Tensor(samples_tra).to(device)
            labels = torch.Tensor(labels).to(device)
            metrics = training.train(samples_tra, labels)
            train_batches_loss.append(metrics[0])
            train_root_mse.append(metrics[1])
            train_score.append(metrics[2])
            pass
        lr_schedule.step()
        t_end = time.time()
        train_time.append(t_end - t_start)

        valid_batches_loss = []
        valid_root_mse = []
        valid_score = []
        t_start_0 = time.time()
        for no_batch, (samples_val, labels) in enumerate(tqdm(data['val_loader'].get_iterator())):
            if len(samples_val) != args.batch_size:
                break
            samples_val = torch.tensor(samples_val, requires_grad=True, dtype=torch.float32, device=device)
            labels = torch.tensor(labels, requires_grad=True, dtype=torch.float32, device=device)
            metrics = training.evaluation(samples_val, labels)
            valid_batches_loss.append(metrics[0])
            valid_root_mse.append(metrics[1])
            valid_score.append(metrics[2])
            pass
        t_end_0 = time.time()
        prompt = "Epoch: {:03d}, Inference Time: {:.4f} seconds."
        print(prompt.format(i, (t_end_0 - t_start_0)))
        valid_time.append(t_end_0 - t_start_0)

        mean_train_loss = np.mean(train_batches_loss)
        mean_train_root_mse = np.mean(train_root_mse)
        mean_train_score = np.mean(train_score)

        mean_val_loss = np.mean(valid_batches_loss)
        mean_val_root_mse = np.mean(valid_root_mse)
        mean_val_score = np.mean(valid_score)

        train_loss.append(mean_train_loss)
        valid_loss.append(mean_val_loss)

        prompt_1 = "Epoch: [{:03d}/{:03d}], Train Loss: {:.3f}, Train Root MSE: {:.3f}, Train Score: {:.3f}."
        prompt_2 = "Epoch: [{:03d}/{:03d}], Valid Loss: {:.3f}, Valid Root MSE: {:.3f}, Valid Score: {:.3f}."
        print(prompt_1.format(args.epochs, i, mean_train_loss, mean_train_root_mse, mean_train_score))
        print(prompt_2.format(args.epochs, i, mean_val_loss, mean_val_root_mse, mean_val_score))

        torch.save(training.model.state_dict(),
                   args.save + args.data + "_exp" + str(args.exp_id) + "_epoch_" + str(i) + "_" +
                   str(round(mean_val_loss, 2)) + ".pth")

        if i % args.patience_epochs == 0:
            current_best_id = np.argmin(valid_loss)
            current_best_loss = valid_loss[current_best_id].item()
            if valid_loss[-1].item() > current_best_loss:
                print("Training is finished now, Epoch: {:03d}.".format(i))
                break

        print("-" * 90)
    print("Average training time: {:.3f} secs/epoch.".format(np.mean(train_time)))
    print("Average inference time: {:.3f} secs/epoch.".format(np.mean(valid_time)))
    best_id = np.argmin(valid_loss)
    training.model.load_state_dict(torch.load(args.save + args.data + "_exp" + str(args.exp_id) + "_epoch_" +
                                              str(best_id+1) + "_" + str(round(valid_loss[best_id], 2)) + ".pth"))
    torch.save(training.model.state_dict(),
               args.save_best + args.data + "_exp" + str(args.exp_id) + "_best_" +
               str(round(valid_loss[best_id], 2)) + '.pth')
    pass


if __name__ == "__main__":
    main()
