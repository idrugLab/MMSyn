import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
from model import MMSyn
from trainer import training_classing, evaluate_test_scores
from dgllife.utils import EarlyStopping
from dataset import TestbedDataset
from torch_geometric.loader import DataLoader
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import random

from metric import accuracy, precision, recall,f1_score, bacc_score, roc_auc, mcc_score, kappa, ap_score

import xgboost as xgb

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True
# 设置随机数种子

def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def get_feature(model, drug1_loader, drug2_loader, device):
    model.eval()
    i = 0
    with torch.no_grad():
        for data in zip(drug1_loader, drug2_loader):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            y = data[0].y.view(-1, 1).long().to(device)
            output1, feature = model(data1, data2)
            if i == 0:
                features = feature
                y_labels = y
            else:
                features = torch.cat((features, feature))
                y_labels = torch.cat((y_labels, y))
            i = i+1
    return features, y_labels

def metric_df(tp, y_pred, y_test):
    acc = accuracy(y_pred, y_test)
    prec = precision(y_pred, y_test)
    rec = recall(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)
    bacc = bacc_score(y_pred, y_test)
    auc_roc = roc_auc(y_pred, y_test)
    mcc = mcc_score(y_pred, y_test)
    kap = kappa(y_pred, y_test)
    ap = ap_score(y_pred, y_test)
    return [tp, acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap]

def mechine_classion(X_train, y_train, X_test, y_test):
    scores = []
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # fpr, tpr, threshold = roc_curve(y_test, y_pre)
    # AUC = auc(fpr, tpr)
    type = 'DeepModel+rf'
    score1 = metric_df(type, y_pred, y_test)
    scores.append(score1)
    clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                      coef0=0.0, shrinking=True, probability=False,
                      tol=1e-3, cache_size=200, class_weight=None,
                      verbose=False, max_iter=-1, decision_function_shape='ovr',
                      random_state=None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # fpr, tpr, threshold = roc_curve(y_test, y_pre)
    # AUC = auc(fpr, tpr)
    # accuracy, precision, recall,f1_score, bacc_score, roc_auc, mcc_score, kappa, ap_score
    type = 'DeepModel+svm'
    score2 = metric_df(type, y_pred, y_test)
    scores.append(score2)
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    # fpr, tpr, threshold = roc_curve(y_test, y_pre)
    # AUC = auc(fpr, tpr)
    type = 'DeepModel+rnn'
    score3 = metric_df(type, y_pred, y_test)
    scores.append(score3)
    scores_df = pd.DataFrame(scores)
    print('The mechine classions are done ...')
    return scores_df

def xgboost_classion(X_train, y_train, X_val, y_val, X_test, y_test):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    scores = []
    y_test = torch.Tensor(y_test)
    xgb_gbc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', learning_rate=0.1, n_estimators=2000,max_depth=4,colsample_bylevel=1,
                                    min_child_weight=8,gamma=1,subsample=0.8,colsample_bytree=1, objective='binary:logitraw',scale_pos_weight=1)
    xgb_gbc.fit(X_train, y_train, eval_set = [(X_val,y_val)], eval_metric = 'auc', early_stopping_rounds=300)
    pre_pro = xgb_gbc.predict_proba(X_test)
    predicted_labels = list(map(lambda x: np.argmax(x), pre_pro))
    predicted_scores = list(map(lambda x: x[1], pre_pro))
    total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
    total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
    total_labels = torch.cat((total_labels, y_test), 0)
    total_labels = total_labels.numpy().flatten()
    total_prelabels = total_prelabels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    tp = 'DeepModel+xgb'
    acc =  metrics.accuracy_score(y_pred=total_prelabels, y_true=total_labels)
    prec = metrics.precision_score(y_pred=total_prelabels, y_true=total_labels)
    rec = metrics.recall_score(y_pred=total_prelabels, y_true=total_labels)
    f1 = metrics.f1_score(y_pred=total_prelabels, y_true=total_labels)
    bacc = metrics.balanced_accuracy_score(y_pred=total_prelabels, y_true=total_labels)
    auc_roc = metrics.roc_auc_score(y_score=total_preds, y_true=total_labels)
    mcc = metrics.matthews_corrcoef(y_pred=total_prelabels, y_true=total_labels)
    kap = metrics.cohen_kappa_score(y2=total_prelabels, y1=total_labels)
    ap = metrics.average_precision_score(y_score=total_preds, y_true=total_labels)
    scores.append([tp, acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap])
    scores_df = pd.DataFrame(scores)
    return scores_df
    

def mechine_scores(modeling, train_batch,  test_batch, criterion, lr, epoch_num, cuda_name, i):

    model_st = modeling.__name__
    print(model_st)
    train_dataA = TestbedDataset(root='data', dataset='train_setA{num}'.format(num=i))
    val_dataA = TestbedDataset(root='data', dataset='val_setA{num}'.format(num=i))
    test_dataA = TestbedDataset(root='data', dataset='test_setA{num}'.format(num=i))
    
    train_dataB = TestbedDataset(root='data', dataset='train_setB{num}'.format(num=i))
    val_dataB = TestbedDataset(root='data', dataset='val_setB{num}'.format(num=i))
    test_dataB = TestbedDataset(root='data', dataset='test_setB{num}'.format(num=i))

    train_loaderA = DataLoader(train_dataA, batch_size=train_batch, shuffle = False)
    val_loaderA = DataLoader(val_dataA, batch_size=train_batch, shuffle = False)
    test_loaderA = DataLoader(test_dataA, batch_size=test_batch, shuffle=False)

    train_loaderB = DataLoader(train_dataB, batch_size=train_batch, shuffle = False)
    val_loaderB = DataLoader(val_dataB, batch_size=train_batch, shuffle = False)
    test_loaderB = DataLoader(test_dataB, batch_size=test_batch, shuffle=False)
    # traioning the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(device)
    deep_model = modeling()
    deep_model = deep_model.to(device)
    optimizer = torch.optim.Adam(deep_model.parameters(), lr=lr)
   
    file_AUCs = 'results/MultiViewNet_matrix{num}.txt'.format(num=i)
    # file_AUCs = 'results_drugcomb/result{num}/matrix.txt'.format(num=i)
    # accuracy, precision, recall,f1_score, bacc_score, roc_auc, mcc_score, kappa, ap_score
    AUCs = ('Epoch\tACC\tPrec\tRec\tF1\tBACC\troc_auc\tmcc\tkappa\tap')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    not_improved_count = 0
    for epoch in range(epoch_num):
        train_loss = training_classing(deep_model, train_loaderA, train_loaderB, optimizer, criterion, device)
        val_loss, y_true, y_pred = evaluate_test_scores(deep_model,  val_loaderA, val_loaderB, criterion, device)
        print('Train epoch: {} \ttrain_loss: {:.6f}'.format(epoch, train_loss))
        print('Train epoch: {} \ttest_loss: {:.6f}'.format(epoch, val_loss))
        AUC = roc_auc(y_pred, y_true)
        if best_auc < AUC:
            best_auc = AUC
            not_improved_count = 0
            type = epoch
            score = metric_df(type, y_pred, y_true)
            save_AUCs(score, file_AUCs)
            torch.save(deep_model.state_dict(), 'results/model_{num}.model'.format(num=i))
        else:
            not_improved_count += 1
        print('best_auc', best_auc)
        if not_improved_count > 50:
            break
    best_deep_model = modeling()
    best_deep_model = best_deep_model.to(device)
    best_deep_model.load_state_dict(torch.load('results_cnn_2/model_{num}.model'.format(num=i)))
    # best_deep_model.load_state_dict(torch.load('202310007_model/results_xgb/done/model_1.model'))
    best_deep_model.eval()
    test_loss, y_true, y_pred = evaluate_test_scores(best_deep_model, test_loaderA, test_loaderB, criterion, device)
    type = 'test'
    score = metric_df(type, y_pred, y_true)
    save_AUCs(score, file_AUCs)
    print('Train epoch: {} \ttest_loss: {:.6f}'.format(epoch, test_loss))
    xgb_train_feature, xgb_train_y = get_feature(best_deep_model, train_loaderA, train_loaderB, device)
    xgb_val_feature, xgb_val_y = get_feature(best_deep_model, val_loaderA, val_loaderB, device)
    xgb_test_feature, xgb_test_y = get_feature(best_deep_model, test_loaderA, test_loaderB, device)
    xgb_train_feature = xgb_train_feature.cpu().numpy()
    xgb_train_y = xgb_train_y.cpu().numpy()
    xgb_val_feature = xgb_val_feature.cpu().numpy()
    xgb_val_y = xgb_val_y.cpu().numpy()
    xgb_test_feature = xgb_test_feature.cpu().numpy()
    xgb_test_y = xgb_test_y.cpu().numpy()
    # auc_xgb_df = xgboost_classion(xgb_train_feature, xgb_train_y, xgb_val_feature, xgb_val_y, xgb_test_feature, xgb_test_y)
    # auc_df = mechine_classion(xgb_train_feature, xgb_train_y, xgb_test_feature, xgb_test_y)
    # auc_xgb_df.to_csv('results_leave_drug_GIN/auc_xgb_df{num}.csv'.format(num=i), index=None) # resuts_espf_gat
    # auc_df.to_csv('results_leave_drug_GIN/auc_mac_df{num}.csv'.format(num=i), index=None)


if __name__ == "__main__":
    setup_seed(42)
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--train_batch', type=int, required=False, default=256, help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=256, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=256, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-5, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=1000, help='Number of epoch')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')

    args = parser.parse_args()

    modeling = MMSyn
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    cuda_name = args.cuda_name
    criterion = nn.BCEWithLogitsLoss()

    for i in range(0, 5):
        mechine_scores(modeling, train_batch, test_batch, criterion, lr, num_epoch, cuda_name, i)

