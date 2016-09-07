from math import log
from math import exp
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import sys


def read_data_by_docid(path):
    with open(path, 'r') as f_r:
        for line in f_r.readlines():
            parts = line.strip().split(' ')
            label = float(parts[0])
            qid = int(parts[1].strip().split(':')[1])
            docid = int(parts[2].strip())
            del parts[0]
            del parts[0]
            del parts[0]

            values = []
            for part in parts:
                values.append(int(part.strip().split(':')[1]))
            data_point = Data_Point(label, qid, values)
            yield docid, data_point

def read_data_by_qid(path):
    with open(path, 'r') as f_r:
        for line in f_r.readline():
            parts = line.strip().split(' ')
            label = float(parts[0])
            qid = int(parts[1].strip().split(':')[1])
            del parts[0]
            del parts[0]

            values = []
            for part in parts:
                values.append(int(part.strip().split(':')[1]))
            data_point = Data_Point(label, qid, values)
            yield data_point

def get_labels(data_list):
    labels = []
    for data_point in data_list.data_points:
        labels.append(data_point.get_label())
    return labels

class Data_Point(object):
    def __init__(self, label, qid, values):
        self.label = label
        self.qid = qid
        self.values = values
    def get_label(self):
        return self.label
    def get_qid(self):
        return self.qid
    def get_features_values(self):
        return self.values

class Data_List(object):
    def __init__(self):
        self.data_points = []

    def append_data_point(self, data_point):
        self.data_points.append(data_point)

    def get_label(self, idx):
        return self.data_points[idx].get_label()

    def get_qid(self, idx):
        return self.data_points[idx].get_qid()

    def get_features_values(self, idx):
        return self.data_points[idx].get_features_values()

class NDCG():
    '''
    Note for calculate the NDCG, often means the same qid(query)
    so data_list is often the same query data
    '''
    def __init__(self, k, len_gains=6, len_discount=5000):
        '''
        :param self:
        :param len_gains: default 6
        :param len_discount: default 5000
        :return:
        '''
        self.k = k
        self.ideal_gains = {}
        self.gains = [ i*i-1 for i in xrange(len_gains)]
        self.discounts = [1/log(i+2, 2) for i in xrange(len_discount)]

    def get_rels(self, data_list):
        rels = []
        for i in xrange(len(data_list.data_points)):
            rels.append(data_list.get_label(i))
        return rels

    def get_discount(self, idx):
        return self.discounts[idx]

    def get_gain(self, idx):
        return self.gains[idx]

    def cal_ideal_dcg(self, rels, topK):
        idx = np.argsort(rels)
        dcg = 0.0
        for i in xrange(topK):
            dcg += self.get_gain(self.rels[idx[i]])*self.get_discount(i)
        return dcg

    def get_dcg(self, rels, topK):
        dcg = 0.0
        for i in xrange(topK):
            dcg += self.get_gain(rels[i])*self.get_discount(i)
        return dcg

    def get_qid2idealgain(self, path):
        '''
        data format : rel qid:val
        :param path:
        :return: {qid, }
        '''
        qid2idealgain = {}
        with open(path, 'r') as f_r:
            last_qid = -1
            for line in f_r.readline():
                parts = line.strip().split(' ')
                qid = int(parts[0])
                rel = float(parts[1].strip().split(':')[1])
                if qid != last_qid:
                    if last_qid == -1:      #first record
                        rels = []
                        last_qid = qid
                        rels.append(rel)
                    else:
                        #calculate ideal DCG using rels
                        size = min(self.k, len(rels))
                        ideal_DCG = self.cal_ideal_dcg(rels, size)
                        qid2idealgain[last_qid] = ideal_DCG
                        rels = []
                        last_qid = qid
                        rels.append(rel)
                else:
                    rels.append(rel)
        return qid2idealgain

    def score(self, data_list, path):
        if len(data_list.data_points) == 0:
            return 0.0

        size = min(self.k, len(data_list.data_points))

        #get relevance from the same qid
        rels = self.get_rels(data_list)

        ideal = 0.0
        qid2ideal = self.get_qid2idealgain(path)

        qid = data_list.get_qid(0)

        if qid2ideal.has_key(qid):
            ideal = qid2ideal[qid]
        else:
            ideal = self.cal_ideal_dcg(rels, size)
            qid2ideal[data_list.get_qid(0)] = ideal

        if ideal<=0.0:
            return 0.0

        return self.get_dcg(data_list, size)/ideal


    def swap_changes(self, path, data_list):
        '''
        :param data_list:
        :return:
        '''
        size = min(self.k, len(data_list.data_points))
        qid = data_list.get_qid(0)

        qid2idealgain = self.get_qid2idealgain(path)

        rels = self.get_rels(data_list)

        if qid2idealgain.has_key(rels):
            ideal = qid2idealgain[rels]
        else:
            ideal = self.cal_ideal_dcg(rels, size)

        mat_changes = [[0]*len(data_list.data_points)]*size

        for i in xrange(size):
            for j in xrange(len(data_list.data_points)):
                if ideal>0:
                    mat_changes[i][j] = mat_changes[j][i] = \
                        (self.get_discount(i)-self.get_discount(j))*\
                        (self.get_gain(rels[i])-self.get_gain(rels[j]))/ideal
        return mat_changes

class Ensemble():
    def __init__(self, num, learning_rate):
        self.num = num
        self.trees = []
        self.learning_rate = learning_rate

    def get_tree(self, idx):
        return self.trees[idx]

    def get_learning_rate(self):
        return self.learning_rate

    def add_tree(self, tree, weight):
        self.trees.append(tree)
        self.learning_rate.append(weight)

    def remove(self):
        self.trees.pop()
        self.learning_rate.pop()

    def get_count(self):
        return len(self.trees)

    def eval(self, point_data):
        result = 0
        for i in xrange(self.get_count()):
            result += self.trees[i].eval(point_data)*self.learning_rate[i]
        return result


class LambdaMART(object):
    '''
    samples = [data_list, data_list ... ]
    '''
    def __init__(self, samples, n_trees, learning_rate, n_leaves, ensemble):
        self.samples = samples
        self.n_trees = n_trees
        self.n_leaves = n_leaves
        self.learning_rate = learning_rate
        self.pesudo_response = []           #different lambda for each iteration
        self.weights = []                   #different for each iteration
        self.ensemble = ensemble
        num_samples = 0
        for sample in samples:
            num_samples += len(sample.data_points)
        self.model_scores = []*num_samples

    def cal_peseudo_response(self, k, samples, path):
        '''
        update weights and lambdas
        :param samples:
        :return:
        '''
        ndcg = NDCG(k)
        result_lambdas = []
        result_weights = []
        for sample in samples:
            peseudo_response = [0]*len(sample.data_points)
            weigths = [0]*len(sample.data_points)
            rels = ndcg.get_rels(sample)
            idx = np.argsort(rels)
            changes_max = ndcg.swap_changes(path, sample)
            for i in xrange(len(sample.data_points)):
                data_point1 = sample.data_points[i]
                mi = idx[i]
                for j in xrange(len(sample.data_points)):
                    if i>self.k and j>self.k:
                        break
                    data_point2 = sample.data_points[j]
                    mj = idx[j]
                    if data_point1.get_label > data_point2.get_label:
                        delta_ndcg = abs(float(changes_max[i][j]))
                        if delta_ndcg>0:
                            rho = 1.0/(1.0+exp(self.model_scores[mi]-self.model_scores[mj]))
                            lambda_val = rho*delta_ndcg
                            peseudo_response[mi] += lambda_val
                            peseudo_response[mj] -= lambda_val
                            delta = rho*(1-rho)*delta_ndcg
                            weigths[mi] += delta
                            weigths[mj] -= delta
            result_lambdas.extend(peseudo_response)
            result_weights.extend(weigths)
        return result_lambdas, result_weights

    def update_scores(self, tree, samples, lambdas, weights):
        '''
        mat two dimension matrix
        '''
        idx_hash = {}

        records = []

        for i in xrange(len(samples)):
            for j in xrange(len(samples[i].data_points)):
                records.append(samples[i].data_points[j])

        for i in xrange(len(records)):
            idx = tree.apply(records[i])
            if idx_hash.has_key(idx):
                idx_hash[idx].append(idx)
            else:
                idx_hash[idx] = []

        new_leaves = []
        model_scores = []

        for leaves_idx in idx_hash.keys():
            s1 = 0.0
            s2 = 0.0
            for samples_idx in idx_hash[leaves_idx]:
                s1 += lambdas[samples_idx]
                s2 += weights[samples_idx]
            if s2 == 0:
                s = 0
            else:
                s = s1/s2
            new_leaves.append(s)

        for leaves_idx in idx_hash.keys():         #this line gets leaves index
            for samples_idx in idx_hash[leaves_idx]:
                model_scores[samples_idx] += self.learning_rate*new_leaves[samples_idx]

        return model_scores

    def train(self, samples, path):
        '''
        algorithm:
        for trees iterator
            for doc in docs
                lambda for per document
                lambda gradient
            for tree in trees              //create tree terminal nodes, not only leaves nodes
                create L terminal nodes for tree,
                using MSE method to determine best split at any node in the regression tree
            for L leaves
                using approximate Newton step find leaf values
            for tree in trees:
                for leave in leaves:
                    f_m = f_m-1 + v*sum()

        :split algorithm: find_best_split(split_node, labels, min_leaf_support):

        OverView Steps:
        calculate lambdas --> regression trees --> update leaves value --> ensemble trees

        :param samples:
        :param topK:
        :param current:
        :return:
        '''
        for i in xrange(self.n_trees):
            peseudo_response, weights = self.cal_peseudo_response(samples, path)
            features_values = samples.get_features_values()
            tree = DecisionTreeRegressor(max_depth=6)
            tree.fit(features_values, peseudo_response)

            self.ensemble.add_tree(tree, self.learning_rate)

            self.model_scores = self.update_scores(tree, samples, peseudo_response, weights)

        '''
        TO DO : using cross validation to rolling back the best trees model
        '''

        return self.ensemble

