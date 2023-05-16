# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from arguments import arg_parse
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator
from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
#from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset

from tqdm import tqdm
import json
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from multiprocessing import cpu_count

class InfoGraph(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1, a=.0):
    super(InfoGraph, self).__init__()
    self.a = a
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, edge_index, batch, num_graphs):
    # batch_size = data.num_graphs
    a = self.a
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)
    #print(l_enc.type())
    # x_rec.require_grad_()
    # print(l_enc.require_grad)

    mode='fd'
    measure='JSD'
    local_global_lossf = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
    if a == 0.0:
        loss_com = local_global_lossf
    else:
        grads_l = torch.autograd.grad(local_global_lossf,
                                    l_enc,
                                    create_graph=True,
                                    only_inputs=True)
        grads_g = torch.autograd.grad(local_global_lossf,
                                    g_enc,
                                    create_graph=True,
                                    only_inputs=True)
        local_global_lossg = local_global_loss_(grads_l[0], grads_g[0], edge_index, batch, measure)
        loss_com = (1-a)*local_global_lossf+a*local_global_lossg
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return loss_com + PRIOR

def my(args,a):
    accuracies = {'svc': []}
    model = InfoGraph(args.hidden_dim, args.num_gc_layers, a).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader)
    # print('===== Before training =====')
    # res = evaluate_embedding(emb, y)
    # accuracies['logreg'].append(res[0])
    # accuracies['svc'].append(res)
    # accuracies['linearsvc'].append(res[2])
    # accuracies['randomforest'].append(res[3])
    best_acc = 0
    best_epoch = 0
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs + 1):
            loss_all = 0
            model.train()
            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            #if epoch % 10 == 0:
                #print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))

            if epoch % args.log_interval == 0:
                model.eval()
                emb, y = model.encoder.get_embeddings(dataloader)
                res = evaluate_embedding(emb, y)
                # accuracies['logreg'].append(res[0])
                accuracies['svc'].append(res)
                if res > best_acc:
                    best_acc = res
                    best_epoch = epoch
                # accuracies['linearsvc'].append(res[2])
                # accuracies['randomforest'].append(res[3])
            pbar.set_postfix({'loss': loss_all})
            pbar.update()
        print(best_acc, best_epoch)
    return best_acc, best_epoch


if __name__ == '__main__':
    args = arg_parse()

    #a = 0.2
    epochs = args.epochs
    #log_interval = args.log_interval
    batch_size = 128
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS).shuffle()
    # print(args)
    #path = osp.join(osp.expanduser('/export/data/rli/Project/new/InfoGraph'), 'datasets')
    dataset_num_features = max(dataset.num_features, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    #print('================')
    #print('lr: {}'.format(lr))
    #print('num_features: {}'.format(dataset_num_features))
    #print('hidden_dim: {}'.format(args.hidden_dim))
    #print('num_gc_layers: {}'.format(args.num_gc_layers))
    #print('================')

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    best_epochs=[]
    runs = 3
    a = args.a
    p = []
    bestepoch = []
    for i in range(0, runs):
        dataset = TUDataset(path, name=DS).shuffle()
        # print(args)
        # path = osp.join(osp.expanduser('/export/data/rli/Project/new/InfoGraph'), 'datasets')
        dataset_num_features = max(dataset.num_features, 1)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        t_1, t_2 = my(args, a)
        p.append(t_1)
        bestepoch.append(t_2)
        print('Now is in', i, ',a is', a)
    p = np.array(p)
    mean.append(p.mean())
    std.append(p.std())
    best_epochs.append(bestepoch)
    with open('logs/log_' + args.DS + '_' + str(args.epochs) + '_' + str(
            args.num_gc_layers), 'a+') as f:
        s = json.dumps([mean, std])
        f.write('{},{},{},{}\n'.format(args.DS, args.epochs, lr, s))
        f.write('\n')