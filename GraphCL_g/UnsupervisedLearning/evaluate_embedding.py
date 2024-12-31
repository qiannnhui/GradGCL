import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

def draw_plot(datadir, DS, embeddings, fname, max_nodes=None):
    return
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    print('fitting TSNE ...')
    x = TSNE(n_components=2).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])

    df['x0'], df['x1'], df['Y'] = x[:,0], x[:,1], y
    sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=5)
    plt.legend()
    plt.savefig(fname)

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def logistic_classify(x, y):

    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):

        # test
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_val.append(acc.item())

    return np.mean(accs_val), np.mean(accs)

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)

def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    ret = np.mean(accuracies)
    return np.mean(accuracies_val), ret

def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, output_dim) 
        self.relu = nn.ReLU()                    
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x


def mlp_classify(x, y, search=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []

    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        model = MLPModel(input_dim=x_train.shape[1], output_dim=len(np.unique(y)))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            # for i in range(len(x_train)):
            #     print("output: ", outputs[i])
            #     print("y_train: ", y_train[i])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            _, preds = torch.max(outputs, 1)
            acc = accuracy_score(y_test.cpu(), preds.cpu()) 
            accuracies.append(acc)

        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if i not in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            _, preds = torch.max(outputs, 1)
            acc_val = accuracy_score(y_test.cpu(), preds.cpu())
            accuracies_val.append(acc_val)

    return np.mean(accuracies_val), np.mean(accuracies)


def evaluate_embedding(embeddings, labels, search=True):

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0

    '''
    _acc_val, _acc = logistic_classify(x, y)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''
    '''
    _acc_val, _acc = svc_classify(x,y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''
    """
    _acc_val, _acc = linearsvc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    """
    '''
    _acc_val, _acc = randomforest_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''

    _acc_val, _acc = mlp_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    #print(acc_val, acc)

    return acc_val, acc

'''
if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
'''
