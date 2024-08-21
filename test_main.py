import utils
import torch
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from GlobalModel import MSTSDNModel


def args_option():
    parser = argparse.ArgumentParser("The required parameters of the model")
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--data', default='FD004', type=str)
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
    parser.add_argument('--save_model', default=None, type=str)
    parser.add_argument('--exp_id', default=None, type=int)
    parser.add_argument('--save', default=None, type=str)
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device)
    data = utils.load_dataset(dataset_name=args.data, rul_early=args.RUL_early,
                              sequence_length=args.num_features, test_sequence=args.test_sequence,
                              batch_size=args.batch_size, valid_batch_size=args.batch_size)

    model = MSTSDNModel(nodes=args.num_nodes,
                        seq=args.num_features, dim_k=args.dim_k,
                        heads=args.t_heads, window_sizes=args.window_size,
                        mlp_ratio=args.mlp_ratio, adj_dim=args.adj_dim,
                        r_mlp_ratio=args.r_mlp_ratio, dropout_rate=args.drop_rate,
                        r_mlp_dropout=args.r_mlp_dropout)

    model.to(device)
    model.load_state_dict(torch.load(args.save_model + args.save))
    model.eval()

    print("Model load successfully." + f" Tested dataset: {args.data}.")
    print('\n' + 80*'-' + '\n', model, '\n' + 80*'-' + '\n',)
    test_samples = torch.tensor(data['test_samples']).to(device)
    test_labels = torch.tensor(data['test_labels']).to(device)
    print("Test labels size: {}.".format(test_labels.shape))
    start_time = time.time()
    with torch.no_grad():
        predictions, mappings, adj = model(test_samples)
    end_time = time.time()
    test_time = end_time - start_time
    print("Estimated RUL size: {}.".format(predictions.shape))
    root_mse, score = utils.get_metrics(predictions, test_labels)
    get_loss = torch.nn.HuberLoss()
    loss = get_loss(predictions, test_labels)
    print("Test Time: {:.20f} secs, Loss: {:.3f}, Root MSE: {:.3f}, Score: {:.3f}.".format(test_time, loss.item(),
                                                                                           root_mse, score))
    predictions = predictions.cpu()
    predictions = np.array(predictions, dtype=np.float32)
    test_labels = test_labels.cpu()
    test_labels = np.array(test_labels, dtype=np.float32)
    pred_error = np.abs((predictions - test_labels))

    return {'predicted rul': predictions, 'actual rul': test_labels, 'extracted features': mappings,
            'raw features': test_samples, 'prediction error': pred_error, 'trained adjacency': adj}


def t_sne_plot(features_mappings, rul_values, args, plot_label: str):
    pre_mappings = features_mappings.cpu()
    mappings_array = np.array(pre_mappings, dtype='float32')
    if len(mappings_array.shape) == 3:
        num_mappings, num_nodes, num_features = pre_mappings.shape
        mappings_array = mappings_array.reshape([num_mappings, num_nodes * num_features])
    rul_array = np.array(rul_values, dtype='float32')
    mappings_array = PCA(n_components=30).fit_transform(mappings_array)
    t_sne = TSNE(n_components=2, learning_rate='auto', random_state=42, init='random',
                 perplexity=39, verbose=0).fit_transform(mappings_array)
    x_min, x_max = t_sne.min(0), t_sne.max(0)
    normal_t_sne = (t_sne - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(20, 16))
    plt.rcParams["font.family"] = 'times new roman'
    plt.grid(True, lw=0.66)
    # YlOrRd
    plt.scatter(normal_t_sne[:, 0], normal_t_sne[:, 1], s=230, c=rul_array,
                label='Dimensional-reduced features', cmap='YlOrRd')
    plt.title(label=plot_label, fontsize=47)
    plt.legend(loc='upper left', fontsize=29)
    plt.xlabel('Dimension 1', fontsize=40)
    plt.xticks(fontproperties='Times New Roman', fontsize=40)
    plt.ylabel('Dimension 2', size=40)
    plt.yticks(fontproperties='Times New Roman', fontsize=40)
    c_bar = plt.colorbar()
    c_bar.ax.tick_params(labelsize=30)
    c_bar.ax.set_ylabel('Remaining useful life', size=40)
    plt.show()
    plot_label = args.data + 'Features Distribution'
    fig.savefig("file address" + plot_label + '.svg', dpi=600, format='svg', bbox_inches='tight')
    pass


def adjacency_heat_map_plot(args, adj, prompts: str, need_adj=False):
    sequence_col = ['S' + str(i) for i in range(1, 29)]
    if 'FD' in args.data:
        sequence_col = [sensor for sensor in sequence_col if
                        sensor not in ['S1', 'S5', 'S6', 'S10', 'S16', 'S18', 'S19',
                                       'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                                       ]]
    adj_title = f"{prompts} Adjacency on {args.data} Dataset."
    adj = adj.cpu()
    adj = np.array(adj, dtype='float32')
    adj_df = pd.DataFrame(adj, index=sequence_col, columns=sequence_col)
    plt.figure(figsize=(24, 18))
    plt.rcParams["font.family"] = 'times new roman'
    plt.title(adj_title, fontsize=42)
    ax = sns.heatmap(adj_df, cmap='bwr', )
    ax.set_yticklabels(sequence_col, rotation=0, fontsize=25)
    ax.set_ylabel('')
    ax.set_xticklabels(sequence_col, rotation=0, fontsize=25)
    c_bar = ax.collections[0].colorbar
    c_bar.ax.tick_params(labelsize=23)
    c_bar.set_label('Similarity Values between Graph Nodes', fontsize=35)
    plt.show()
    if need_adj:
        heat_map = ax.get_figure()
        heat_map.savefig("file address" + f'{args.data}_{prompts}_Adjacency' + '.svg', dpi=512, format='svg',
                         bbox_inches='tight')


def plot_result_engine(y_true, y_pred, pred_error, set_name, fig_size: tuple, engine_id=None, title=None):
    pred_error = list(pred_error.squeeze(-1))
    id_pred_error = list(range(len(pred_error)))
    fig = plt.figure(figsize=fig_size)
    plt.rcParams["font.family"] = 'times new roman'
    line = plt.gca()
    line.spines['top'].set_linewidth(1.5)
    line.spines['bottom'].set_linewidth(1.5)
    line.spines['left'].set_linewidth(1.5)
    line.spines['right'].set_linewidth(1.5)
    plt.grid(True, lw=2)
    plt.rcParams['axes.linewidth'] = 0.8
    plt.plot(y_true, 'red', linewidth=4, linestyle='-', marker='o', markersize=10)
    plt.plot(y_pred, 'navy', linewidth=4, linestyle='-', marker='^', markersize=10)
    plt.bar(id_pred_error, pred_error, width=0.8, fc='lawngreen')
    plt.xlabel('Time Cycle', fontsize=40, )
    plt.xticks(fontproperties='Times New Roman', fontsize=30)
    plt.ylabel('Remaining Useful Life (RUL)', fontsize=40,)
    plt.yticks(fontproperties='Times New Roman', fontsize=30)
    plt.title(title, fontsize=47,)
    labels = plt.legend(['Actual RUL', 'Predicted RUL', 'Absolute Prediction Error'],
                        loc='upper right', fontsize=36,).get_texts()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.show()
    if engine_id is not None:
        fig.savefig("file address"
                    + set_name + f'{engine_id}.svg', dpi=600, format='svg', bbox_inches='tight')
    else:
        fig.savefig("file address"
                    + set_name + '.svg', dpi=600, format='svg', bbox_inches='tight')


if __name__ == '__main__':
    args = args_option()
    prompt = f'RUL Estimation on {args.data} Dataset.'
    pred_rul = main(args)
    if 'FD' in args.data:
        plot_result_engine(y_true=pred_rul['actual rul'], y_pred=pred_rul['predicted rul'],
                           pred_error=pred_rul['prediction error'],
                           set_name=args.data, title=prompt)
        t_sne_plot(features_mappings=pred_rul['extracted features'], rul_values=pred_rul['predicted rul'],
                   plot_label=f'Regression Performance on {args.data} Dataset.', args=args)
        adjacency_heat_map_plot(args, adj=pred_rul['trained adjacency'], prompts='Original')
    elif 'DS' in args.data:
        rul_pred = pred_rul['predicted rul']
        rul_labels = pred_rul['actual rul']
        pred_error = pred_rul['prediction error']

        engine_cycle_list = [0, 0, 0, 0]
        engine_id_list = ['0', '0', '0', '0']
        i = 0
        start = 0
        end = engine_cycle_list[i]

        while True:
            set_name = args.data + engine_id_list[i]
            title = f"RUL Prediction on #{engine_id_list[i]} Engine of {args.data} Dataset."
            rul_pred_plot = rul_pred[start:end, ]
            rul_labels_plot = rul_labels[start:end, ]
            pred_error_plot = pred_error[start:end, ]

            plot_result_engine(y_true=rul_labels_plot, y_pred=rul_pred_plot, pred_error=pred_error_plot,
                               set_name=set_name, engine_id=engine_id_list[i], title=title, fig_size=(0, 0))
            if i == len(engine_cycle_list) - 1:
                break
            start += engine_cycle_list[i]
            i += 1
            end += engine_cycle_list[i]
        t_sne_plot(features_mappings=pred_rul['extracted features'], rul_values=rul_pred,
                   plot_label=f'Regression Performance on {args.data} Dataset.', args=args)
        adjacency_heat_map_plot(args, adj=pred_rul['trained adjacency'], prompts='Original')
    else:
        raise ValueError('Tne name of data file is not correct !')

