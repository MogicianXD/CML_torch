import numpy
import torch
from torch import nn
import torch.nn.functional as F
from util import *
from BaseModel import BaseModel


class CML(BaseModel):
    def __init__(self,
                 n_users,
                 n_items,
                 use_cuda=True,
                 embed_dim=20,
                 features=None,
                 margin=1.5,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=0.1
                 ):
        """

        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature loss(default: None)
        :param margin: hinge loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        :param hidden_layer_dim: the size of feature projector's hidden layer (default: 128)
        :param dropout_rate: the dropout rate between the hidden layer to final feature projection layer
        :param feature_l2_reg: feature loss weight
        :param feature_projection_scaling_factor: scale the feature projection before compute l2 loss. Ideally,
               the scaled feature projection should be mostly within the clip_norm
        :param use_rank_weight: whether to use rank weight
        :param use_cov_loss: use covariance loss to discourage redundancy in the user/item embedding
        """
        super(CML, self).__init__(use_cuda)
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.margin = margin
        if features is not None:
            self.features = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            self.features = None

        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight

        self.user_embeddings = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.embed_dim)
        for emb in [self.user_embeddings, self.item_embeddings]:
            nn.init.normal_(emb.weight.data, std=1 / (self.embed_dim ** 0.5))

        if self.features:
            self.mlp = nn.Sequential(
                nn.Linear(self.features.shape[-1], self.hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_layer_dim, self.embed_dim)
            )

        if self.features:
            self.item_embeddings.weight.data = self._feature_projection()

    def _feature_projection(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """
        # feature loss
        if self.features is not None:
            # fully-connected layer
            output = self.mlp(self.features) * self.feature_projection_scaling_factor
            # projection to the embedding
            return clip_by_norm(output, self.clip_norm)

    def _feature_loss(self):
        """
        :return: the l2 loss of the distance between items' their embedding and their feature projection
        """
        loss = torch.tensor(0, dtype=torch.float32)
        feature_projection = self._feature_projection()
        if feature_projection is not None:
            # apply regularization weight
            loss += torch.sum((self.item_embeddings.weight.data - feature_projection) ** 2) * self.feature_l2_reg
        return loss

    def _covariance_loss(self):
        X = torch.cat((self.item_embeddings.weight.data, self.user_embeddings.weight.data), 0)
        n_rows = X.shape[0]
        X -= X.mean(0)
        cov = torch.matmul(X.transpose(-2, -1), X) / n_rows
        loss = cov.sum() - cov.trace()
        return loss * self.cov_loss_weight

    def _embedding_loss(self, X):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # user embedding (N, K)
        users = self.user_embeddings(X[:, 0])

        # positive item embedding (N, K)
        pos_items = self.item_embeddings(X[:, 1])
        # positive item to user distance (N)
        pos_distances = torch.sum((users - pos_items) ** 2, 1)

        # negative item embedding (N, K, W)
        neg_items = self.item_embeddings(X[:, 2:]).transpose(-2, -1)
        # distance to negative items (N x W)
        distance_to_neg_items = torch.sum((users.unsqueeze(-1) - neg_items) ** 2, 1)

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_item_distances = distance_to_neg_items.min(1)[0]

        # compute hinge loss (N)
        distance = pos_distances - closest_negative_item_distances + self.margin
        loss_per_pair = F.relu(distance)

        if self.use_rank_weight:
            # indicator matrix for impostors (N x W)
            impostors = (pos_distances.unsqueeze(-1) - distance_to_neg_items + self.margin) > 0
            # approximate the rank of positive item by (number of impostor / W per user-positive pair)
            rank = impostors.float().mean(1) * self.n_items
            # apply rank weight
            loss_per_pair *= torch.log(rank + 1)

        # the embedding loss
        loss = loss_per_pair.sum()

        return loss

    def loss(self, X):
        """
        :return: the total loss = embedding loss + feature loss
        """
        X = X.to(self.device)
        feature_loss = self._feature_loss()
        loss = self._embedding_loss(X) + feature_loss
        if self.use_cov_loss:
            loss += self._covariance_loss()
        return loss, feature_loss

    def clip_by_norm_op(self):
        return [clip_by_norm(self.user_embeddings.weight.data, self.clip_norm),
                clip_by_norm(self.item_embeddings.weight.data, self.clip_norm)]

    def fit(self, data, task):
        self.train()
        c = []
        for input_batch in data:
            self.optimizer.zero_grad()
            loss, feature_loss = task(input_batch)
            c.append(loss.item())
            if self.features:
                loss += feature_loss / self.n_items
            loss.backward()
            self.optimizer.step()
            self.clip_by_norm_op()
        return np.mean(c)

    def item_scores(self, score_user_ids):
        score_user_ids = score_user_ids.to(self.device)
        # (N_USER_IDS, 1, K)
        user = self.user_embeddings(score_user_ids).unsqueeze(1)
        # (1, N_ITEM, K)
        item = self.item_embeddings.weight.data.unsqueeze(0)
        # score = minus distance (N_USER, N_ITEM)
        return -torch.sum((user - item) ** 2, 2)


    def test_rank(self, data, **kwargs):
        self.eval()
        data, train_pos, test_pos = data
        total = len(data.dataset)
        top = kwargs['topk']
        HR, MRR, NDCG = [0] * len(top), [0] * len(top), [0] * len(top)
        with torch.no_grad():
            for input_batch in data:
                preds = self.item_scores(input_batch)
                ranks = []
                for uid, pred in zip(input_batch, preds):
                    pred[train_pos[uid.item()]] = -1e6
                    ranks.append(min([(pred > pred[t]).sum() for t in test_pos[uid.item()]]) + 1)
                ranks = preds.new_tensor(ranks)
                for k in range(len(top)):
                    rank_ok = (ranks <= top[k])
                    HR[k] += rank_ok.sum().item()
                    MRR[k] += mrr(ranks[rank_ok])
                    NDCG[k] += ndcg(ranks[rank_ok])
        return [np.array(metric, dtype=float) / total for metric in [HR, MRR, NDCG]]