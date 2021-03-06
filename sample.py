"""
Sampling strategies to generate negative examples from knowledge graphs
with an open-world assumption

From scikit-kge (author: Maximiliam Nickel)
"""

from copy import deepcopy
from collections import defaultdict as ddict
from numpy.random import randint
from random import choice


class Sampler(object):

    def __init__(self, n, modes=[0,1], ntries=100):
        self.n = n
        self.modes = modes
        self.ntries = ntries

    def sample(self, xys):
        res = []
        for x, _ in xys:
            for _ in range(self.n):
                mode = choice(self.modes)
                t = self._sample(x, mode)
                if t is not None:
                    res.append(t)
        return res


class SOSampler(Sampler):
    def __init__(self, n, xs, sz):
        super(SOSampler, self).__init__(n)
        self.sz = sz
        self.so = set([(s,o) for s,o,p in xs])

    def _sample(self, x, mode):
        nex = list(x)
        res = None
        for _ in range(self.ntries):
            nex[mode] = randint(self.sz[mode])
            nex[2] = -1
            if tuple(nex[0:2]) not in self.so:
                res = (tuple(nex), -1.0)
                break
        return res


class RandomModeSampler(Sampler):
    """
    Sample negative triples randomly
    """

    def __init__(self, n, modes, xs, sz):
        super(RandomModeSampler, self).__init__(n, modes)
        self.xs = set(xs)
        self.sz = sz

    def _sample(self, x, mode):
        nex = list(x)
        res = None
        for _ in range(self.ntries):
            nex[mode] = randint(self.sz[mode])
            if tuple(nex) not in self.xs:
                res = (tuple(nex), -1.0)
                break
        return res


class RandomSampler(Sampler):

    def __init__(self, n, xs, sz):
        super(RandomSampler, self).__init__(n)
        self.xs = set(xs)
        self.sz = sz

    def _sample(self, x, mode):
        res = None
        for _ in range(self.ntries):
            nex = (randint(self.sz[0]),
                   randint(self.sz[1]),
                   randint(self.sz[2]))
            if nex not in self.xs:
                res = (nex, -1.0)
                break
        return res


class CorruptedSampler(RandomSampler):

    def __init__(self, n, xs, sz, type_index):
        super(CorruptedSampler, self).__init__(n, xs, sz)
        self.type_index = type_index

    def _sample(self, x, mode):
        nex = list(deepcopy(x))
        res = None
        for _ in range(self.ntries):
            if mode == 2:
                nex[2] = randint(len(self.type_index))
            else:
                k = x[2]
                n = len(self.type_index[k][mode])
                nex[mode] = self.type_index[k][mode][randint(n)]
            if tuple(nex) not in self.xs:
                res = (tuple(nex), -1.0)
                break
        if res is None:
            res = super(CorruptedSampler,self)._sample(x,mode)
        return res


class LCWASampler(RandomModeSampler):
    """
    Sample negative examples according to the local closed world assumption
    """

    def __init__(self, n, modes, xs, sz):
        super(LCWASampler, self).__init__(n, modes, xs, sz)
        self.counts = ddict(int)
        for s, o, p in xs:
            self.counts[(s, p)] += 1

    def _sample(self, x, mode):
        nex = list(deepcopy(x))
        res = None
        for _ in range(self.ntries):
            nex[mode] = randint(self.sz[mode])
            if self.counts[(nex[0], nex[2])] > 0 and tuple(nex) not in self.xs:
                res = (tuple(nex), -1.0)
                break
        if res is None:
            res = super(LCWASampler,self)._sample(x,mode)
        return res


def type_index(xs):
    index = ddict(lambda: {0: set(), 1: set()})
    for i, j, k in xs:
        index[k][0].add(i)
        index[k][1].add(j)
    return {k: {0: list(v[0]), 1: list(v[1])} for k, v in index.items()}