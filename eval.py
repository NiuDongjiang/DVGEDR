import torch
# from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from torch.optim import Adam
from torch_geometric.data import DataLoader
#from torch.utils.data import DataLoader
from torch_geometric.utils import dropout_adj
import dgl
from tqdm import tqdm
import numpy as np
import torch.optim as optim

def train_epochs(train_dataset, train_dataset1, test_dataset, test_dataset1, model, args,seed):
    num_workers = 2
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False,
                              num_workers=num_workers)

    train_loader1 = DataLoader(train_dataset1, args.batch_size, shuffle=False,
                               num_workers=num_workers)


    test_size = 1024
    test_loader = DataLoader(test_dataset, test_size, shuffle=False,
                             num_workers=num_workers)
    test_loader1 = DataLoader(test_dataset1, test_size, shuffle=False,
                              num_workers=num_workers)
    model.to(args.device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    start_epoch = 1
    pbar = range(start_epoch, args.epochs + start_epoch)
    best_epoch, best_auc, best_aupr = 0, 0, 0
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, train_loader1, args.device, epoch)
        if epoch % args.valid_interval == 0:
            roc_auc, aupr = evaluate_metric(model, test_loader, test_loader1, args.device, epoch)
            print("epoch {}".format(epoch), "train_loss {0:.4f}".format(train_loss),
                  "roc_auc {0:.4f}".format(roc_auc), "aupr {0:.4f}".format(aupr))
            if roc_auc > best_auc:
                best_epoch, best_auc, best_aupr = epoch, roc_auc, aupr

    print("best_epoch {}".format(best_epoch), "best_auc {0:.4f}".format(best_auc), "aupr {0:.4f}".format(best_aupr))

    return best_auc, best_aupr


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, loader1, device, epoch):
    model.train()
    total_loss = 0
    pbar = loader
    test = False
    for data1, data2 in tqdm(zip(pbar, loader1), desc='train_epoch{}'.format(epoch), ncols=100):
        optimizer.zero_grad()
        true_label1 = data1.to(device)
        true_label2 = data2.to(device)
        predict = model(true_label1, true_label2,epoch,test)
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(predict, true_label1.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


def evaluate_metric(model, loader, loader1, device, epoch):
    model.eval()
    pbar = loader
    roc_auc, aupr = None, None
    test = True
    y_true_l=[]
    y_score_l=[]
    for data1, data2 in tqdm(zip(pbar, loader1), desc='test_epoch{}'.format(epoch), ncols=100):
        data1 = data1.to(device)
        pyg_data_list1 = data1.to_data_list()
        data2 = data2.to(device)
        with torch.no_grad():
            out = model(data1, data2, epoch,test)

        y_true = data1.y.view(-1).cpu().tolist()
        y_score = out.cpu().numpy().tolist()
        for i in y_true:
            y_true_l.append(i)
        for i in y_score:
            y_score_l.append(i)

    fpr, tpr, _ = metrics.roc_curve(y_true_l, y_score_l)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y_true_l, y_score_l)
    aupr = metrics.auc(recall, precision)
    torch.cuda.empty_cache()

    return roc_auc, aupr
