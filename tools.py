import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from numpy import argsort
from collections import defaultdict as ddict
import os
import csv
from tqdm import tqdm
from numpy.fft import fft, ifft
import math
from keras.callbacks import Callback
from keras.engine.topology import Layer
from scipy.sparse import coo_matrix
import tensorflow as tf



def to_tensor(triples, n_relations, n_instances, order="sop"):
    assert len(order) == 3 and "s" in order and "o" in order and "p" in order
    triples = np.array(triples)
    if order != "sop":
        idx = [order.index("s"), order.index("o"), order.index("p")]
        triples = triples[:, idx]

    X = []
    for i in range(n_relations):
        r_idx = np.where(triples[:, 2] == i)[0]
        r_triples = triples[r_idx]
        rows = list(r_triples[:, 0])
        cols = list(r_triples[:, 1])
        data = [1] * len(r_idx)
        r_matrix = coo_matrix((data, (rows, cols)),
                              shape=(n_instances, n_instances))
        X.append(r_matrix)
    return X


def huang2003_layers(n_feats, n_classes):
    n_feats = float(n_feats)
    n_classes = float(n_classes)
    n_layer1 = math.sqrt((n_classes + 2) * n_feats) + 2 * math.sqrt(n_feats / (n_classes + 2))
    n_layer2 = n_classes * math.sqrt(n_feats / (n_classes + 2))
    return [int(n_layer1), int(n_layer2)]


class RankCallback(Callback):
    def __init__(self, eval, lp, freq=10, patience=2):
        self.eval = eval
        self.freq = freq
        self.best_fmrr = 0
        self.patience = patience
        self.patience_count = 0
        self.lp = lp
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}):
        if ((epoch % self.freq) == 0) and epoch:
            pos, fpos = self.eval.positions(self.lp)
            fmrr = ranking_scores(pos, fpos)
            self.val_loss = fmrr
            if fmrr > self.best_fmrr:
                print("fmrr new best=%f, previous=%f" % (fmrr, self.best_fmrr))
                self.best_fmrr = fmrr
                self.patience_count = 0
                self.best_weights = self.lp.model.get_weights()
            else:
                self.patience_count += 1
                print("patience=%d" % self.patience_count)
                if self.patience_count > self.patience:
                    self.lp.model.set_weights(self.best_weights)
                    self.model.stop_training = True


def ccorr(a, b):
    return ifft(np.conj(fft(a)) * fft(b)).real


def lp_scores(mdl, xs, ys):
    scores = mdl.predict_proba(xs)
    pr, rc, _ = precision_recall_curve(ys, scores)
    roc = roc_auc_score(ys, scores)
    f1 = 2 * (np.multiply(pr, rc)) / (pr + rc)
    return auc(rc, pr), roc, np.max(np.nan_to_num(f1))


def ranking_scores(pos, fpos):
    hpos = [p for k in pos.keys() for p in pos[k]['head']]
    tpos = [p for k in pos.keys() for p in pos[k]['tail']]
    fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
    ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]
    print_pos(
        np.array(hpos),
        np.array(fhpos), prefix="head")
    print_pos(
        np.array(tpos),
        np.array(ftpos), prefix="tail")
    fmrr = print_pos(
        np.array(hpos + tpos),
        np.array(fhpos + ftpos), prefix="all")
    return fmrr


def print_pos(pos, fpos, prefix=""):
    mrr, mean_pos, hits = compute_scores(pos, hits=[10, 5, 3, 1])
    fmrr, fmean_pos, fhits = compute_scores(fpos, hits=[10, 5, 3, 1])
    print(
        "%s: MRR = %.3f/%.3f, Mean Rank = %.3f/%.3f, Hits@10 = %.3f/%.3f Hits@5 = %.1f/%.1f Hits@3 = %.1f/%.1f Hits@1 = %.1f/%.1f" %
        (prefix, mrr, fmrr, mean_pos, fmean_pos, hits[0], fhits[0], hits[1], fhits[1], hits[2], fhits[2], hits[3],
         fhits[3])
    )
    return fmrr


def compute_scores(pos, hits=[10]):
    mrr = np.mean(1.0 / pos)
    mean_pos = np.mean(pos)
    hits = [np.mean(pos <= hits_i).sum() * 100 for hits_i in hits]
    return mrr, mean_pos, hits


class FilteredRankingEval(object):
    def __init__(self, xs, true_triples, neval=-1, verbose=False):
        self.verbose = verbose
        idx = ddict(list)
        tt = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        at = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        self.neval = neval
        self.sz = len(xs)
        for s, o, p in xs:
            idx[p].append((s, o))

        for s, o, p in true_triples:
            tt[p]['os'][s].append(o)
            tt[p]['ss'][o].append(s)

        self.idx = dict(idx)
        self.tt = dict(tt)

        self.neval = {}
        for p, sos in self.idx.items():
            if neval == -1:
                self.neval[p] = -1
            else:
                self.neval[p] = np.int(np.ceil(neval * len(sos) / len(xs)))

    def scores_o(self, mdl, s, p):
        triples = []
        for o in range(mdl.n_instances):
            triples.append((s, o, p))
        return mdl.predict_proba(triples)

    def scores_s(self, mdl, o, p):
        triples = []
        for s in range(mdl.n_instances):
            triples.append((s, o, p))
        return mdl.predict_proba(triples)

    def rank_o(self, mdl, s, p, o):
        scores_o = self.scores_o(mdl, s, p).flatten()
        if scores_o.shape[0] == 1 and scores_o.shape[0] < scores_o.shape[1]:
            scores_o = np.ravel(scores_o)
        sortidx_o = argsort(scores_o)[::-1]
        rank = np.where(sortidx_o == o)[0][0] + 1

        rm_idx = self.tt[p]['os'][s]
        rm_idx = [i for i in rm_idx if i != o]
        scores_o[rm_idx] = -np.Inf
        sortidx_o = argsort(scores_o)[::-1]
        frank = np.where(sortidx_o == o)[0][0] + 1
        return rank, frank

    def rank_s(self, mdl, s, p, o):
        scores_s = self.scores_s(mdl, o, p).flatten()
        if scores_s.shape[0] == 1 and scores_s.shape[0] < scores_s.shape[1]:
            scores_s = np.ravel(scores_s)
        sortidx_s = argsort(scores_s)[::-1]
        rank = np.where(sortidx_s == s)[0][0] + 1

        rm_idx = self.tt[p]['ss'][o]
        rm_idx = [i for i in rm_idx if i != s]
        scores_s[rm_idx] = -np.Inf
        sortidx_s = argsort(scores_s)[::-1]
        frank = np.where(sortidx_s == s)[0][0] + 1
        return rank, frank

    def positions(self, mdl):
        pos = {}
        fpos = {}

        if hasattr(self, 'prepare_global'):
            self.prepare_global(mdl)

        for p, sos in self.idx.items():
            if p in self.tt:
                ppos = {'head': [], 'tail': []}
                pfpos = {'head': [], 'tail': []}

                if hasattr(self, 'prepare'):
                    self.prepare(mdl, p)

                for s, o in sos[:self.neval[p]]:
                    rank, frank = self.rank_o(mdl, s, p, o)
                    ppos['tail'].append(rank)
                    pfpos['tail'].append(frank)
                    rank, frank = self.rank_s(mdl, s, p, o)
                    ppos['head'].append(rank)
                    pfpos['head'].append(frank)
                pos[p] = ppos
                fpos[p] = pfpos

        return pos, fpos

class PairwiseEval(FilteredRankingEval):
    def positions(self, mdl):
        self.all_s = set()
        self.all_o = set()
        for so_list in self.idx.values():
            for s, o in so_list:
                self.all_s.add(s)
                self.all_o.add(o)
        self.o_rank_values = {p: {s: {} for s in self.tt[p]["os"].keys()} for p in self.tt.keys()}
        self.s_rank_values = {p: {o: {} for o in self.tt[p]["ss"].keys()} for p in self.tt.keys()}
        self.o_filtered_rank_values = {p: {s: {} for s in self.tt[p]["os"].keys()} for p in self.tt.keys()}
        self.s_filtered_rank_values = {p: {o: {} for o in self.tt[p]["ss"].keys()} for p in self.tt.keys()}
        n = mdl.n_instances

        with mdl.sess.as_default():
            if self.verbose:
                print("Predicting objects")
                pbar = tqdm(total=len(self.all_s))
            for s in self.all_s:
                for p in self.tt.keys():
                    ranks = mdl.rank.eval(
                        feed_dict={mdl.input_s: np.full((n, 1), s), mdl.input_o: np.arange(n).reshape(-1, 1),
                                   mdl.input_p: np.full((n, 1), p)})
                    if s in self.tt[p]["os"]:
                        for o in self.tt[p]["os"][s]:
                            rank = np.where(ranks == o)[1][0] + 1
                            self.o_rank_values[p][s][o] = rank
                        for i, (o, rank) in enumerate(sorted(self.o_rank_values[p][s].items(), key=lambda x: x[1])):
                            self.o_filtered_rank_values[p][s][o] = rank - i
                if self.verbose: pbar.update(1)
            if self.verbose: pbar.close()

            if self.verbose:
                print("Predicting subjects")
                pbar = tqdm(total=len(self.all_o))
            for o in self.all_o:
                for p in self.tt.keys():
                    ranks = mdl.rank.eval(
                        feed_dict={mdl.input_s: np.arange(n).reshape(-1, 1), mdl.input_o: np.full((n, 1), o),
                                   mdl.input_p: np.full((n, 1), p)})
                    if o in self.tt[p]["ss"]:
                        for s in self.tt[p]["ss"][o]:
                            rank = np.where(ranks == s)[1][0] + 1
                            self.s_rank_values[p][o][s] = rank
                        for i, (s, rank) in enumerate(sorted(self.s_rank_values[p][o].items(), key=lambda x: x[1])):
                            self.s_filtered_rank_values[p][o][s] = rank - i
                if self.verbose: pbar.update(1)
            if self.verbose: pbar.close()
            return super(PairwiseEval, self).positions(mdl)

    def prepare(self, mdl, p):
        pass

    def rank_o(self, mdl, s, p, o):
        return self.o_rank_values[p][s][o], self.o_filtered_rank_values[p][s][o]

    def rank_s(self, mdl, s, p, o):
        return self.s_rank_values[p][o][s], self.s_filtered_rank_values[p][o][s]

class MultilabelEval(FilteredRankingEval):
    def positions(self, mdl):
        self.all_s = set()
        self.all_o = set()
        for so_list in self.idx.values():
            for s, o in so_list:
                self.all_s.add(s)
                self.all_o.add(o)
        self.o_rank_values = {p: {s: {} for s in self.tt[p]["os"].keys()} for p in self.tt.keys()}
        self.s_rank_values = {p: {o: {} for o in self.tt[p]["ss"].keys()} for p in self.tt.keys()}
        self.o_filtered_rank_values = {p: {s: {} for s in self.tt[p]["os"].keys()} for p in self.tt.keys()}
        self.s_filtered_rank_values = {p: {o: {} for o in self.tt[p]["ss"].keys()} for p in self.tt.keys()}
        n = mdl.n_instances

        with mdl.sess.as_default():
            if self.verbose:
                print("Predicting objects")
                pbar = tqdm(total=len(self.all_s))
            for s in self.all_s:
                ranks = mdl.rank.eval(
                    feed_dict={mdl.input_s: np.full((n, 1), s), mdl.input_o: np.arange(n).reshape(-1, 1)})
                for p in self.tt.keys():
                    if s in self.tt[p]["os"]:
                        for o in self.tt[p]["os"][s]:
                            rank = np.where(ranks[p] == o)[0][0] + 1
                            self.o_rank_values[p][s][o] = rank
                        for i, (o, rank) in enumerate(sorted(self.o_rank_values[p][s].items(), key=lambda x: x[1])):
                            self.o_filtered_rank_values[p][s][o] = rank - i
                if self.verbose: pbar.update(1)
            if self.verbose: pbar.close()

            if self.verbose:
                print("Predicting subjects")
                pbar = tqdm(total=len(self.all_o))
            for o in self.all_o:
                ranks = mdl.rank.eval(
                    feed_dict={mdl.input_s: np.arange(n).reshape(-1, 1), mdl.input_o: np.full((n, 1), o)})
                for p in self.tt.keys():
                    if o in self.tt[p]["ss"]:
                        for s in self.tt[p]["ss"][o]:
                            rank = np.where(ranks[p] == s)[0][0] + 1
                            self.s_rank_values[p][o][s] = rank
                        for i, (s, rank) in enumerate(sorted(self.s_rank_values[p][o].items(), key=lambda x: x[1])):
                            self.s_filtered_rank_values[p][o][s] = rank - i
                if self.verbose: pbar.update(1)
            if self.verbose: pbar.close()
            return super(MultilabelEval, self).positions(mdl)

    def prepare(self, mdl, p):
        pass

    def rank_o(self, mdl, s, p, o):
        return self.o_rank_values[p][s][o], self.o_filtered_rank_values[p][s][o]

    def rank_s(self, mdl, s, p, o):
        return self.s_rank_values[p][o][s], self.s_filtered_rank_values[p][o][s]


def decode_triple(triple, order="sop"):
    s_i, o_i, p_i = order.index("s"), order.index("o"), order.index("p")
    return triple[s_i], triple[o_i], triple[p_i]

def load_data(input, order="sop"):
    train_path = os.path.join(input, "train.txt")
    test_path = os.path.join(input, "test.txt")
    valid_path = os.path.join(input, "valid.txt")
    entity2id_path = os.path.join(input, "entity2id.txt")
    relation2id_path = os.path.join(input, "relation2id.txt")

    entity2id = {}
    for ent, id in csv.reader(open(entity2id_path, "rb"), delimiter="\t"):
        entity2id[ent] = int(id)
    n_ents = len(entity2id)
    print("%d entities loaded" % n_ents)

    relation2id = {}
    for rel, id in csv.reader(open(relation2id_path, "rb"), delimiter="\t"):
        relation2id[rel] = int(id)
    n_rels = len(relation2id)
    print("%d relations loaded" % n_rels)

    ss, oo, pp = [], [], []
    for triple in csv.reader(open(train_path, "rb"), delimiter="\t"):
        s, o, p = decode_triple(triple, order)
        ss.append(entity2id[s])
        oo.append(entity2id[o])
        pp.append(relation2id[p])
    train_triples = zip(ss, oo, pp)

    ss, oo, pp = [], [], []
    for triple in csv.reader(open(test_path, "rb"), delimiter="\t"):
        s, o, p = decode_triple(triple, order)
        ss.append(entity2id[s])
        oo.append(entity2id[o])
        pp.append(relation2id[p])
    test_triples = zip(ss, oo, pp)

    ss, oo, pp = [], [], []
    for triple in csv.reader(open(valid_path, "rb"), delimiter="\t"):
        s, o, p = decode_triple(triple, order)
        ss.append(entity2id[s])
        oo.append(entity2id[o])
        pp.append(relation2id[p])
    valid_triples = zip(ss, oo, pp)

    return train_triples, test_triples, valid_triples, entity2id, relation2id
