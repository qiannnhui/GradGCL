import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch_sparse import SparseTensor
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
from torch import optim
from torch.nn.parameter import Parameter
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from model import *
from arguments import arg_parse
from evaluate_embedding import evaluate_embedding
from torch_geometric.transforms import Constant
import pdb
import logging
from torch.autograd import Variable
from copy import deepcopy

#LOG_FORMAT = "%(levelname)s - %(message)s"
#DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
#logging.basicConfig(filename='Accuracy.txt',level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
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
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR

class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
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
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)        
        return y

    def loss_cal(self, x, x_aug, args):
        T = args.T
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def gen_ran_output(data, model, vice_model, args):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)           
    z2 = vice_model(data.x, data.edge_index, data.batch, data.num_graphs)
    return z2

def gen_ran_output_eval(dataloader, model, vice_model, args):
    for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0, torch.ones_like(param.data) * param.data.std()).to(
                device)
    z2,_ = vice_model.encoder.get_embeddings_grad(dataloader)
    return z2

class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)


class TwoAugUnsupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)

    def __len__(self):
        return len(self.dataset)

def get_grad(embed, loss):
    grads = torch.autograd.grad(loss,
                                        embed,
                                        create_graph=True,
                                        only_inputs=True)
    grad = grads[0]
    embedding_dim = args.hidden_dim * args.num_gc_layers
    proj_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True),
                                   nn.Linear(embedding_dim, embedding_dim)).to(device)
    grad = proj_head(grad)
    #print('after',grad.size())
    return grad

def my(args,a):
    accuracies = {'val': [], 'test': []}
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    vice_model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #print('================')
    #print('lr: {}'.format(lr))
    #print('num_features: {}'.format(dataset_num_features))
    #print('hidden_dim: {}'.format(args.hidden_dim))
    #print('num_gc_layers: {}'.format(args.num_gc_layers))
    #print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """
    best_acc = 0.0
    best_epoch = 0
    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs + 1):
            loss_all = 0
            model.train()
            for data in dataloader:
                optimizer.zero_grad()
                node_num, _ = data.x.size()
                data = data.to(device)
                x2 = gen_ran_output(data, model, vice_model, args)
                x1 = model(data.x, data.edge_index, data.batch, data.num_graphs)
                # loss_aug = model.loss_cal(x2, x1)
                # loss = loss_aug
                lossf = model.loss_cal(x2, x1,args)
                if a == 0.0:
                    loss = lossf
                else:
                    grads1, grads2 = get_grad(x1, lossf), get_grad(x2, lossf)
                    lossg = model.loss_cal(grads2, grads1, args)
                # a = 0.4
                    loss = (1 - a) * lossf + a * lossg
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            #if epoch % 10 == 0:
                #print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

            if epoch % args.log_interval == 0:
                model.eval()
                emb, y = model.encoder.get_embeddings(dataloader_eval)
                acc_val, acc = evaluate_embedding(emb, y)
                #accuracies['val'].append(acc_val)
                #accuracies['test'].append(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
            pbar.set_postfix({'loss': loss_all})
            pbar.update()
        print(best_acc, best_epoch)
        #print(grads1.size())
        #print(x2.size())
    return best_acc


if __name__ == '__main__':
    args = arg_parse()
    #setup_seed(args.seed)
    device = torch.device(args.device)
    #accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    #log_interval = 20
    batch_size = args.batch_size
    lr = args.lr
    #a = args.a
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    import numpy as np
    import matplotlib.pyplot as plt

    # args = arg_parse()
    mean = []
    std = []
    a = args.a
    # eta = 0.1
    p = []

    for i in range(1, 6):
        dataset = TUDataset(path, name=DS).shuffle()
        dataset_eval = TUDataset(path, name=DS).shuffle()
        # print(len(dataset))
        # print(len(dataset_eval))
        # print(dataset.get_num_feature())
        try:
            dataset_num_features = dataset.get_num_feature()
        except:
            dataset_num_features = 1
        p.append(my(args, a))
        print('Now is in', i, ',a is', a)
    p = np.array(p)
    mean.append(p.mean())
    std.append(p.std())
    with open('logs/log_' + args.DS + '_' + str(args.eta), 'a+') as f:
        s = json.dumps([mean,std])
        f.write('{},{},{},{},{}\n'.format(args.DS, args.eta, epochs, lr, s))
        f.write('\n')

