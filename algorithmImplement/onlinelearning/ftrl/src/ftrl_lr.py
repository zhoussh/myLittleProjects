from csv import DictReader
from math import log, exp, sqrt
import random

def read_data_line(path, dimension):
    '''
    get data from csv file
    :param path:
    :param num_weights:
    :return:
    '''
    for i, row in enumerate(DictReader(open(path))):
        id = row['id']
        del row['id']

        y = 0.0
        if row['click'] == '1':
            y = 1.0
        del row['click']

        # row]'date'] format is 'YYMMDDHH', e.g. : 14091123
        day = int(row['hour'][4:6])
        row['hour'] = row['hour'][6:]

        xs = []

        for key in row:
            val = row[key]
            idx = abs(hash(key + '_' + val))%dimension
            xs.append(idx)

        yield i, day, id, xs, y

def bounded_sigmoid(x):
    return 1.0/(1.0+exp(-max(min(x,35.0), -35.0)))

def bounded_logloss(p, y):
    '''
    :param p: prediction value
    :param y: true value
    :return:
    '''
    p = max(min(p, 1.0-10e-15), 10e-15)
    return -log(p) if y == 1.0 else -log(1.0 - p)

class Ftrl(object):

    def __init__(self, alpha, beta, l1, l2, dimension, ways):
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2

        self.dimension = dimension
        self.ways = ways

        self.n = [0.0] * dimension
        self.z = [0.0] * dimension
        self.w = [0.0] * dimension

    def _get_index(self, xs):
        '''
        item for the polynomial, likely
        sum(xixj) + sum(xi) + 0
        :param xs:
        :return:
        '''
        yield 0             #bias

        for x in xs:        # one degree items for the polynomial
            yield x

        if self.ways:
            dimension = self.dimension
            l = len(xs)

            xs = sorted(xs)

            for i in xrange(l):
                for j in xrange(i+1, l):
                    yield abs(hash(str(xs[i]) + '_' + str(xs[j])))%dimension

    def update(self, xs, p, y):
        '''
        This function is
        :param xs:
        :param p:
        :param y:
        :return:
        '''
        alpha = self.alpha
        n = self.n
        z = self.z
        w = self.w

        diff = p - y
        for i in self._get_index(self, xs):
            grad = diff * xs[i]
            rho = (sqrt(n[i]+grad*grad)-sqrt(n[i]))/alpha
            z[i] = z[i]+grad-rho*w[i]
            n[i] = n[i]+grad*grad


    def predict(self, xs):
        '''
        :param xs:
        :return: probability, p(y=1|x;w) = sigmoid(wTx)
        '''
        alpha = self.alpha
        beta = self.beta
        l1 = self.l1
        l2 = self.l2

        n = self.n
        z = self.z
        w = {}

        wTx = 0

        for i in self._get_index(xs):
            sign_z = -1.0 if z<0 else 1.0
            if abs(z[i]) < l1:
                w[i] = 0
            else:
                w[i] = (sign_z(z[i])*l1 - z[i])/((beta+sqrt(n[i]))/alpha+l2)
            wTx += w[i]*xs[i]

        self.w = w
        return bounded_sigmoid(wTx)

    def train_validate(self, path, num_round, holdafter, holdout):
        alpha = self.alpha
        beta = self.beta
        l1 = self.l1
        l2 = self.l2
        dimension = self.dimension
        ways = self.ways

        for t in xrange(num_round):
            loss = 0.0
            count = 0

            for i, day, id, xs, true_y in read_data_line(path, self.dimension):
                pred = self.predict(xs)

                #how to cross  validate?
                #1. split data into train data[day 0-holdafter] and validate[holdafter:]
                    #validatioin part
                if (holdafter and day>holdafter) or (holdout and i%holdout==0):
                    loss += bounded_logloss(pred, true_y)
                    count += 1
                else:
                    #train part
                    self.update(xs, pred, true_y)

        return self.w