from calendar import c
from cmath import inf
import math
from model.core import Model
import numpy as np


refusal = -1
class DecisionTree(Model):
    def __init__(self):
        self.pos_weights = 0.0
        self.neg_weights = 0.0
        self.feature = -1
        self.cut = 0.0
        self.left_category = 0
        self.right_category = 0
        self.entropy = 0.0
        return

    def _calc_gain(self, pos_weights, total_weights):
        pos = pos_weights / total_weights
        neg = 1 - pos
        cond_entropy0 = 0.0
        if pos != 0:
            cond_entropy0 += -pos * math.log2(pos)
        if neg != 0:
            cond_entropy0 += -neg * math.log2(neg)
        pos = (self.pos_weights - pos_weights) / (1 - total_weights)
        neg = 1 - pos
        cond_entropy1 = 0.0
        if pos > 0:
            cond_entropy1 += -pos * math.log2(pos)
        if neg > 0:
            cond_entropy1 += -neg * math.log2(neg)
        return self.entropy-(total_weights * cond_entropy0 + (1 - total_weights) * cond_entropy1)

    def _calc_inherant_value(self, total_weights):
        res = 0.0
        if total_weights > 0:
            res += -total_weights * math.log2(total_weights)
        if total_weights < 1:
            res += -(1-total_weights) * math.log2(1-total_weights)
        return res

    def _calc_gain_ratio(self, pos_weights, total_weights):

        return self._calc_gain(pos_weights, total_weights) / self._calc_inherant_value(total_weights)

    def _calc_gini(self, pos_weights, total_weights):
        pos = pos_weights / total_weights
        neg = 1 - pos
        gini0 = 1 - pos ** 2 - neg ** 2
        pos = (self.pos_weights - pos_weights) / (1 - total_weights)
        neg = 1 - pos
        gini1 = 1 - pos ** 2 - neg ** 2
        return total_weights * gini0 + (1 - total_weights) * gini1


    def _calc_weights(self, train_Y, weight):
        for i in range(train_Y.shape[0]):
            if train_Y[i] == 1:
                self.pos_weights += weight[i]
            else:
                self.neg_weights += weight[i]
        if self.pos_weights > 0:
            self.entropy += -self.pos_weights * math.log2(self.pos_weights)
        if self.neg_weights > 0:
            self.entropy += -self.neg_weights * math.log2(self.neg_weights)
        return

    def _evaluate_feature_i(self, train_X, train_Y, weights, i):
        pairs = []
        for j in range(train_X.shape[0]):
            pairs.append([train_X[j, i], train_Y[j], weights[j]]) # (feature, label, weight)
        pairs.sort()

        pos_weights = 0.0
        total_weights = 0.0

        max_neg_cond_entropy = -1e18
        cut = 0.0
        category_l = 0.0
        category_r = 0.0
        for j in range(len(pairs) - 1):
            pair = pairs[j]
            total_weights += pair[2]
            if pair[1] > 0.5:
                pos_weights += pair[2] 
            if pairs[j][0] == pairs[j + 1][0]:
                continue
            neg_cond_entropy = -self._calc_gini(pos_weights, total_weights)
            if neg_cond_entropy > max_neg_cond_entropy:
                max_neg_cond_entropy = neg_cond_entropy
                cut = (pairs[j][0] + pairs[j + 1][0]) / 2
                category_l = 1 if pos_weights * 2 > total_weights else 0
                category_r = 1 if (self.pos_weights - pos_weights) * 2 > (1 - total_weights) else 0

                

        return (max_neg_cond_entropy, cut, category_l, category_r) # 

    def blocking(self, train_X, train_Y, weights, feature, cut, c):
        res = []
        for i in range(train_X.shape[0]):
            feat = train_X[i, feature]
            if feat <= cut:
                res.append(c)
            else:
                res.append(1 - c)
        res = np.array(res)
        err_rate = (res != train_Y).dot(weights)
        if err_rate > 0.5:
            return True
        else:
            return False


    def train_with_weights(self, train_X, train_Y, weights):
        self._calc_weights(train_Y, weights)
        feature = -1
        max_neg_cond_entropy = -inf
        cut = 0.0
        category_l, category_r = 0.0, 0.0
        for i in range(train_X.shape[1]):
            if i == refusal:
                continue
            ce, c, cate_l, cate_r = self._evaluate_feature_i(train_X, train_Y, weights, i)
            if ce > max_neg_cond_entropy:
                (feature, max_neg_cond_entropy, cut, category_l, category_r) = (i, ce, c, cate_l, cate_r)
        self.feature = feature
        self.cut = cut
        self.left_category = category_l
        self.right_category = category_r

    def evaluate(self, X):
        res = []
        for i in range(X.shape[0]):
            feat = X[i, self.feature]
            if feat <= self.cut:
                res.append(self.left_category)
            else:
                res.append(self.right_category)
        return np.array(res)

    def predict(self, X):
        return self.evaluate(X)