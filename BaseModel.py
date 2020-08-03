import math
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from util import *


class BaseModel(nn.Module):
    def __init__(self, use_cuda=True):
        super(BaseModel, self).__init__()
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def train_test(self, train_task, test_task, train_data, valid_data, test_data=None, reload=False,
                   n_epochs=10, lr=0.01, n_metric=2, ref=-1, savepath=None, small_better=None,
                   **kwargs):
        if reload and os.path.exists(savepath):
            print('reload...')
            self.load_state_dict(torch.load(savepath))
        if small_better is None:
            small_better = [False] * n_metric
        best_epoch = [-1] * n_metric
        best_metrics = [1e5 if small else 0 for small in small_better]
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        change = False
        for epoch in range(n_epochs):
            avgc = self.fit(train_data, train_task)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))

            metric = test_task(valid_data, **kwargs)
            for i, m in enumerate(metric):
                if kwargs.get('topk'):
                    m = m[-1]
                if (best_metrics[i] < m) ^ small_better[i]:
                    best_metrics[i], best_epoch[i] = m, epoch
                    change = True
                    if savepath and i in [ref, n_metric + ref]:
                        torch.save(self.state_dict(), savepath)
            print('best_epoch', best_epoch)
            print('valid', metric)
            # if test_data and change:
            #     metric = test_task(test_data, **kwargs)
            #     print('test', metric)
            #     change = False


    def fit(self, data, task):
        self.train()
        c = []
        for input_batch in data:
            self.optimizer.zero_grad()
            # input_batch = torch.tensor(mask(input_batch)).to(self.device)
            # target_batch = torch.LongTensor(target_batch).to(self.device)
            cost = task(input_batch)
            c.append(cost.item())
            cost.backward()
            self.optimizer.step()
        return np.mean(c)

    def fit_bpr(self, input_batch):
        preds = self.forward(input_batch)
        return bpr(preds)

    def fit_nll(self, input_batch):
        input_X, groundtruth = input_batch
        preds = self.forward(input_X)
        cost = F.cross_entropy(preds, groundtruth)
        return cost

    def fit_nll_neg(self, input_batch, epsilon=1e-9):
        preds = torch.sigmoid(self.forward(input_batch))
        cost = - torch.log(preds[:, 0] + epsilon).sum() - torch.log(1 - preds[:, 1:] + epsilon).sum()
        return cost / preds.shape[0]

    def fit_mse(self, input_batch):
        input_X, groundtruth = input_batch
        groundtruth = groundtruth.to(self.device).float()
        preds = self.forward(input_X)
        cost = F.mse_loss(preds, groundtruth)
        return cost

    def test_rank(self, data, **kwargs):
        self.eval()
        total = len(data.dataset)
        if kwargs.get('topk'):
            top = kwargs['topk']
        else:
            top = [5, 10, 20]
        HR, NDCG = [0] * len(top), [0] * len(top)
        with torch.no_grad():
            for input_batch, target_batch in data:
                preds = self.forward(input_batch)
                ranks = (preds > torch.diag(preds[target_batch])).sum(0) + 1
                for k in range(len(top)):
                    rank_ok = (ranks <= top[k])
                    HR[k] += rank_ok.sum().item()
                    NDCG[k] += ndcg(ranks[rank_ok])
        return np.array(HR, dtype=float) / total, np.array(NDCG, dtype=float) / total

    def test_rank_with_neg(self, data, **kwargs):
        self.eval()
        total = len(data.dataset)
        if kwargs.get('topk'):
            top = kwargs['topk']
        else:
            top = [5, 10, 20]
        HR, NDCG = [0] * len(top), [0] * len(top)
        with torch.no_grad():
            for input_batch in data:
                preds = self.forward(input_batch)
                ranks = (preds > preds[:, 0].unsqueeze(-1)).sum(1) + 1
                for k in range(len(top)):
                    rank_ok = (ranks <= top[k])
                    HR[k] += rank_ok.sum().item()
                    NDCG[k] += ndcg(ranks[rank_ok])
        return np.array(HR, dtype=float) / total, np.array(NDCG, dtype=float) / total

    def test_classify(self, data, **kwargs):
        self.eval()
        preds = torch.zeros(0, device=self.device)
        with torch.no_grad():
            for input_batch in data:
                pred = torch.sigmoid(self.forward(input_batch))
                preds = torch.cat((preds, pred), -2)
        preds = preds.cpu()
        gt = torch.zeros_like(preds).cpu()
        gt[:, 0] = 1
        gt = gt.flatten()
        preds = preds.flatten()
        AUC = roc_auc_score(gt, preds)
        RMSE = math.sqrt(mean_squared_error(gt, preds))
        return AUC, RMSE

    def test_regression(self, data, **kwargs):
        self.eval()
        total = len(data.dataset)
        MAE, RRSE = 0, 0
        x2_sum, x_sum = 0, 0
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(data):
                preds = self.forward(input_batch)
                target_batch = target_batch.to(self.device)
                balance = preds - target_batch
                MAE += torch.abs(balance).sum()
                RRSE += (balance ** 2).sum()
                x2_sum += (target_batch ** 2).sum()
                x_sum += target_batch.sum()
        var = x2_sum / total - (x_sum / total) ** 2
        RRSE = math.sqrt(RRSE.item() / total) / var.item()
        return MAE.item() / total, RRSE

