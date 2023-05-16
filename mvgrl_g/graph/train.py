import numpy as np
import argparse
import json
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#from graph.dataset import load
from dataset import load
import os
from multiprocessing import cpu_count

def argument():
    parser = argparse.ArgumentParser(description='mvgrl')
    # data source params
    parser.add_argument('--DS', type=str, default='MUTAG', help='Name of dataset.')
    parser.add_argument('--measure', type=str, default='JSD', help='Type of loss measure')
    parser.add_argument('--control', type=int, default=10, help='control factor, threads = cpu_num/control')

    # training params
    parser.add_argument('--device', type=str, default='cpu', help='cuda:{}, default:cpu.')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--log_interval', type=int, default=20, help='Interval between two evaluations.')

    # model params
    parser.add_argument('--num_layers', type=int, default=3, help='Number of graph convolution layers before each pooling.')
    parser.add_argument('--hid_dim', type=int, default=32, help='Hidden layer dimensionalities.')
    parser.add_argument('--a', type=float, default=0, help='Gradient weight')

    #args = parser.parse_args()

    return parser.parse_args()

class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNLayer(in_ft, n_h).to(args.device))
        for __ in range(num_layers - 1):
            self.layers.append(GCNLayer(n_h, n_h).to(args.device))

    def forward(self, feat, adj, mask):
        h_1 = self.layers[0](feat, adj)
        h_1g = torch.sum(h_1, 1)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, adj)
            h_1g = torch.cat((h_1g, torch.sum(h_1, 1)), -1)
        return h_1, h_1g


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


class Model(nn.Module):
    def __init__(self, n_in, n_h, num_layers):
        super(Model, self).__init__()
        self.mlp1 = MLP(1 * n_h, n_h)
        self.mlp2 = MLP(num_layers * n_h, n_h)
        self.gnn1 = GCN(n_in, n_h, num_layers)
        self.gnn2 = GCN(n_in, n_h, num_layers)

    def forward(self, adj, diff, feat, mask):
        lv1, gv1 = self.gnn1(feat, adj, mask)
        lv2, gv2 = self.gnn2(feat, diff, mask)

        lv1 = self.mlp1(lv1)
        lv2 = self.mlp1(lv2)

        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)

        return lv1, gv1, lv2, gv2

    def embed(self, feat, adj, diff, mask):
        __, gv1, __, gv2 = self.forward(adj, diff, feat, mask)
        return (gv1 + gv2).detach()


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_global_loss_(l_enc, g_enc, batch, measure, mask):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(args.device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(args.device)
    msk = torch.ones((num_nodes, num_graphs)).to(args.device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    for idx, m in enumerate(mask):
        msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.

    res = torch.mm(l_enc, g_enc.t()) * msk

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).to(args.device)
    neg_mask = torch.ones((num_graphs, num_graphs)).to(args.device)
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def train(args,a):
    nb_epochs = args.epochs
    batch_size = args.batch_size
    patience = 20
    lr = args.lr
    l2_coef = 0.0
    #args.hid_dim = 512

    adj, diff, feat, labels, num_nodes = load(args.DS)

    feat = torch.FloatTensor(feat).to(args.device)
    diff = torch.FloatTensor(diff).to(args.device)
    adj = torch.FloatTensor(adj).to(args.device)
    labels = torch.LongTensor(labels).to(args.device)

    ft_size = feat[0].shape[1]
    max_nodes = feat[0].shape[0]

    model = Model(ft_size, args.hid_dim, args.num_layers)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    model.to(args.device)

    cnt_wait = 0
    best = 1e9

    itr = (adj.shape[0] // batch_size) + 1
    with tqdm(total=nb_epochs, desc='(T)') as pbar:
        for epoch in range(nb_epochs):
            epoch_loss = 0.0
            train_idx = np.arange(adj.shape[0])
            np.random.shuffle(train_idx)

            for idx in range(0, len(train_idx), batch_size):
                model.train()
                optimiser.zero_grad()

                batch = train_idx[idx: idx + batch_size]
                mask = num_nodes[idx: idx + batch_size]

                lv1, gv1, lv2, gv2 = model(adj[batch], diff[batch], feat[batch], mask)

                lv1 = lv1.view(batch.shape[0] * max_nodes, -1)
                lv2 = lv2.view(batch.shape[0] * max_nodes, -1)
                #print(lv1.type())
                #x_rec.require_grad_()
                #print(lv1.require_grad)

                batch = torch.LongTensor(np.repeat(np.arange(batch.shape[0]), max_nodes)).to(args.device)

                loss1f = local_global_loss_(lv1, gv2, batch, args.measure, mask)
                loss2f = local_global_loss_(lv2, gv1, batch, args.measure, mask)
                #print(lo.type())
                # x_rec.require_grad_()
                # print(x_rec.require_grad)
                if a != 0.0:
                    # print(a)
                    grads_l1 = torch.autograd.grad(loss1f, lv1, create_graph=True, only_inputs=True)

                    grads_g2 = torch.autograd.grad(loss1f, gv2, create_graph=True, only_inputs=True)

                    #grads_l2 = torch.autograd.grad(loss2f, lv2, create_graph=True, only_inputs=True)

                    #grads_g1 = torch.autograd.grad(loss2f, gv1, create_graph=True, only_inputs=True)
                    #torch.cat((lv1, grads_l1),0)
                    loss1g = local_global_loss_(grads_l1[0], grads_g2[0], batch, args.measure, mask)
                    #loss2g = local_global_loss_(grads_l2[0], grads_g1[0], batch, args.measure, mask)
                    loss1 = (1 - a) * loss1f + a * loss1g
                    #loss2 = (1 - a) * loss2f + a * loss2g
                    loss2 = loss2f
                elif a == 0:
                    # print(a)
                    loss1 = loss1f
                    loss2 = loss2f
                # loss3 = global_global_loss_(gv1, gv2, args.measure)
                loss = loss1 + loss2 #+ loss3
                epoch_loss += loss
                loss.backward()
                optimiser.step()

            epoch_loss /= itr

            # print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, epoch_loss))

            if epoch_loss < best:
                best = epoch_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), f'{args.DS}-{args.device}.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                break
            pbar.set_postfix({'loss': epoch_loss})
            pbar.update()
    model.load_state_dict(torch.load(f'{args.DS}-{args.device}.pkl'))

    features = feat.to(args.device)
    adj = adj.to(args.device)
    diff = diff.to(args.device)
    labels = labels.to(args.device)

    embeds = model.embed(features, adj, diff, num_nodes)

    #x = embeds.cpu().numpy()
    #y = labels.cpu().numpy()
    x, y = np.array(embeds), np.array(labels)

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    print(np.mean(accuracies), np.std(accuracies))
    return np.mean(accuracies),best_t


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = argument()
    device = torch.device(args.device)

    if args.device == 'cpu':
        cpu_num = cpu_count()  # 自动获取最大核心数目
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        control = args.control
        #control = 10
        thread = int(cpu_num / control)
        torch.set_num_threads(thread)

    import numpy as np
    import matplotlib.pyplot as plt

    # args = arg_parse()
    mean = []
    std = []
    best_epochs = []
    runs = 3
    a = args.a
    p = []
    best_epoch = []
    for i in range(0, runs):
        t_1, t_2 = train(args, a)
        p.append(t_1)
        best_epoch.append(t_2)
        print('Now is in', i, ',a is', a)
    p = np.array(p)
    mean.append(p.mean())
    std.append(p.std())
    best_epochs.append(best_epoch)

    with open('logs/log_' + args.DS + '_' + str(args.epochs), 'a+') as f:
        s = json.dumps([mean, std, best_epochs])
        f.write('{},{},{},{}\n'.format(args.DS, args.epochs, args.lr, s))
        f.write('\n')

