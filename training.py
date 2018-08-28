from keras.layers import Dense, Dropout, Embedding, Activation, Merge, Input, merge, Flatten, Lambda, \
    Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Conv3D, MaxPooling3D, Reshape, LocallyConnected1D, LocallyConnected2D, \
    AveragePooling2D, Multiply, Dot, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adagrad, RMSprop
import numpy as np
from collections import defaultdict
import random
from sample import type_index, LCWASampler, CorruptedSampler, RandomModeSampler, SOSampler
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.constraints import maxnorm, nonneg, unitnorm
from keras.regularizers import l1, l2
import warnings


class ModelCheckpointNoException(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        try:
            super(ModelCheckpointNoException,self).on_epoch_end(epoch, logs)
        except Exception as e:
            warnings.warn("cound not checkpoint model: Error=%s"%(e.message))


class KGEM(object):
    def fit(self, train_triples, valid_triples):
        pass

    def predict(self, test_triples):
        pass

    def save_model(self, path):
        pass

    def close(self):
        pass

    def triple_score(self, s, o, p):
        pass


class PairwiseMargin(KGEM):
    def __init__(self, n_dim, n_relations, n_instances, activation="relu",
                 dropout=0.25, n_neg=10, sample_mode="lcwa", epochs=100, constraint=1, regularizer=0.001,
                 rank_callback=False, margin=0.5,
                 batch_size=200, combination="simple", initialization=None, n_channels=1, verbose=0, layers=None,
                 patience=10):

        random.seed(88)
        # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.activation = activation
        self.dropout = dropout
        self.n_neg = n_neg
        self.sample_mode = sample_mode
        self.training_epochs = epochs
        self.batch_size = batch_size
        self.combination = combination
        self.initialization = initialization
        self.verbose = verbose
        self.margin = margin
        self.layers = layers
        self.n_channels = n_channels
        self.dimensions = n_dim
        self.rel_dimensions = n_dim
        self.n_relations = n_relations
        self.n_instances = n_instances
        self.patience = patience
        self.rank_callback = rank_callback
        self.constraint = maxnorm(constraint, axis=1) if constraint != 0.0 else None
        self.regularizer = l1(regularizer) if regularizer else None
        self.border_mode = "valid"
        self.sigm_act = Activation("sigmoid")
        self.callbacks = []
        self.callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False))
        self.callbacks.append(
            EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='auto'))
        self.callbacks.append(
            ModelCheckpointNoException(filepath="/tmp/%s.hdf5" % self.__class__.__name__, verbose=1, save_best_only=True))

    def create_sampler(self, triples, sz):
        if self.sample_mode == "so":
            sampler = SOSampler(self.n_neg, triples, sz)
        elif self.sample_mode == 'corrupted':
            ti = type_index(triples)
            sampler = CorruptedSampler(self.n_neg, triples, sz, ti)
        elif self.sample_mode == 'random':
            sampler = RandomModeSampler(self.n_neg, [0, 1], triples, sz)
        elif self.sample_mode == 'lcwa':
            sampler = LCWASampler(self.n_neg, [0, 1], triples, sz)
        return sampler

    def triple_score(self, s, o, p):
        subject_emb = self.ent_embeddings(s)
        object_emb = self.ent_embeddings(o)
        relation_emb = self.rel_embeddings(p)
        return self.triple_score_embs(subject_emb, object_emb, relation_emb)

    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        pass

    def create_variables(self):
        self.ent_embeddings = Embedding(input_dim=self.n_instances, output_dim=self.dimensions, trainable=True,
                                        W_constraint=self.constraint)
        self.rel_embeddings = Embedding(input_dim=self.n_relations, output_dim=self.rel_dimensions, trainable=True,
                                        W_constraint=self.constraint)

    def create_model(self, train_triples=None, valid_triples=None):
        self.input_s = input_s = Input(shape=(1,), dtype="int32", name="input_s")
        self.input_o = input_o = Input(shape=(1,), dtype="int32", name="input_o")
        self.input_p = input_p = Input(shape=(1,), dtype="int32", name="input_p")

        self.input_neg_s = input_neg_s = Input(shape=(1,), dtype="int32", name="input_neg_s")
        self.input_neg_o = input_neg_o = Input(shape=(1,), dtype="int32", name="input_neg_o")
        self.input_neg_p = input_neg_p = Input(shape=(1,), dtype="int32", name="input_neg_p")

        self.create_variables()

        x_pos = self.triple_score(input_s, input_o, input_p)
        x_neg = self.triple_score(input_neg_s, input_neg_o, input_neg_p)
        predictions = merge([x_pos, x_neg], mode="concat")

        def max_pairwise_margin(y_true, y_pred):
            y_pos = y_pred[:, 0]
            y_neg = y_pred[:, 1]
            return tf.reduce_sum(tf.maximum(0., -y_pos + y_neg + self.margin))

        self.model = Model(input=[input_s, input_o, input_p, input_neg_s, input_neg_o, input_neg_p], output=predictions)
        self.model.compile(loss=max_pairwise_margin, optimizer=Adam())
        self.model.summary()

        self.pred = x_pos
        _, self.rank = tf.nn.top_k(tf.transpose(self.pred), k=self.n_instances)

    def batch_generator(self, train_triples, sampler):
        while 1:
            x_s, x_o, x_p, x_ns, x_no, x_np, Y = self.convert_data(train_triples, sampler)
            idx = np.arange(Y.shape[0])
            np.random.shuffle(idx)
            for i in range(idx.shape[0] // self.batch_size):
                batch_idx = idx[i * self.batch_size:(i + 1) * self.batch_size]
                yield [x_s[batch_idx], x_o[batch_idx], x_p[batch_idx], x_ns[batch_idx], x_no[batch_idx],
                       x_np[batch_idx]], Y[batch_idx]

    def convert_data(self, triples, sampler=None):
        r_negs = sampler.sample(zip(triples, [1] * len(triples)))
        neg_triples = [(s, o, p) for (s, o, p), y in r_negs]

        x_s = np.array([[s] * self.n_neg for s, o, p in triples], dtype="int32").reshape((-1, 1))
        x_o = np.array([[o] * self.n_neg for s, o, p in triples], dtype="int32").reshape((-1, 1))
        x_p = np.array([[p] * self.n_neg for s, o, p in triples], dtype="int32").reshape((-1, 1))

        x_ns = np.array([s for s, o, p in neg_triples], dtype="int32").reshape((-1, 1))
        x_no = np.array([o for s, o, p in neg_triples], dtype="int32").reshape((-1, 1))
        x_np = np.array([p for s, o, p in neg_triples], dtype="int32").reshape((-1, 1))

        y = np.ones((len(triples) * self.n_neg, 1))

        return x_s, x_o, x_p, x_ns, x_no, x_np, y

    def fit_generator(self, train_triples, valid_triples):
        self.create_model(train_triples, valid_triples)

        sz = (self.n_instances, self.n_instances, self.n_relations)
        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        validation_x_s, validation_x_o, validation_p, validation_x_ns, validation_x_no, validation_np, validation_y = \
            self.convert_data(valid_triples, sampler=v_sampler)

        self.model.fit_generator(self.batch_generator(train_triples, sampler),
                                 samples_per_epoch=len(train_triples) * (self.n_neg + 1),
                                 epochs=self.training_epochs, nb_worker=24, max_q_size=10, verbose=self.verbose,
                                 callbacks=self.callbacks,
                                 validation_data=(
                                     [validation_x_s, validation_x_o, validation_p, validation_x_ns, validation_x_no,
                                      validation_np], validation_y))

    def fit(self, train_triples, valid_triples):
        self.create_model(train_triples, valid_triples)
        sz = (self.n_instances, self.n_instances, self.n_relations)

        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        x_s, x_o, x_p, x_ns, x_no, x_np, Y = self.convert_data(train_triples, sampler=sampler)

        validation_x_s, validation_x_o, validation_p, validation_x_ns, validation_x_no, validation_np, validation_y = \
            self.convert_data(valid_triples, sampler=v_sampler)

        self.model.fit([x_s, x_o, x_p, x_ns, x_no, x_np], Y,
                       batch_size=self.batch_size, epochs=self.training_epochs,
                       verbose=self.verbose, shuffle=True, callbacks=self.callbacks,
                       validation_data=(
                           [validation_x_s, validation_x_o, validation_p, validation_x_ns, validation_x_no,
                            validation_np],
                           validation_y))

    def predict_proba(self, triples):
        ss = np.array([s for s, o, p in triples], dtype="int32").reshape((-1, 1))
        oo = np.array([o for s, o, p in triples], dtype="int32").reshape((-1, 1))
        pp = np.array([p for s, o, p in triples], dtype="int32").reshape((-1, 1))

        with self.sess.as_default():
            return self.pred.eval(feed_dict={self.input_s: ss, self.input_o: oo, self.input_p: pp})

    def predict(self, triples):
        return self.predict_proba(triples) > 0.5

    def save_model(self, path):
        try:
            self.model.save(path + ".h5")
        except Exception as e:
            raise Warning("cound not checkpoint model to %s: Error=%s"%(path,e.message))



class OptimizeRank(PairwiseMargin):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        pass

    def joint_triple_score(self, ss, os, ps):
        all_subjects_emb = self.ent_embeddings(ss)
        all_objects_emb = self.ent_embeddings(os)
        all_relations_emb = self.rel_embeddings(ps)
        return self.joint_triple_score_embs(all_subjects_emb, all_objects_emb, all_relations_emb)

    def joint_triple_score_embs(self, subject_emb, object_emb, relation_emb):
        scores = []
        for i in range(0, self.n_neg + 1):
            column_i = Lambda(lambda x: x[:, i])
            s_i = column_i(subject_emb)
            o_i = column_i(object_emb)
            p_i = column_i(relation_emb)
            scores.append(self.triple_score_embs(s_i, o_i, p_i))
        scores = Concatenate()(scores)
        return scores

    def create_model(self, train_triples=None, valid_triples=None):
        self.act = Activation("sigmoid")
        self.joint_input_s = joint_input_s = Input(shape=(1 + self.n_neg,), dtype="int32", name="joint_input_s")
        self.joint_input_o = joint_input_o = Input(shape=(1 + self.n_neg,), dtype="int32", name="joint_input_o")
        self.joint_input_p = joint_input_p = Input(shape=(1 + self.n_neg,), dtype="int32", name="joint_input_p")

        self.input_s = input_s = Input(shape=(1,), dtype="int32", name="input_s")
        self.input_o = input_o = Input(shape=(1,), dtype="int32", name="input_o")
        self.input_p = input_p = Input(shape=(1,), dtype="int32", name="input_p")

        self.create_variables()

        predictions = self.joint_triple_score(joint_input_s, joint_input_o, joint_input_p)

        def max_pairwise_margin(y_true, y_pred):
            y_pos = tf.reshape(y_pred[:, 0], [-1, 1])
            y_pos = tf.tile(y_pos, multiples=[1, self.n_neg])
            y_neg = y_pred[:, 1:]
            loss = tf.reduce_sum(tf.maximum(0., -y_pos + y_neg + self.margin))
            return tf.div(loss, self.n_neg)

        self.model = Model(input=[joint_input_s, joint_input_o, joint_input_p], output=predictions)
        self.model.compile(loss=max_pairwise_margin, optimizer=Adam())
        self.model.summary()

        self.pred = self.triple_score(input_s, input_o, input_p)
        _, self.rank = tf.nn.top_k(tf.transpose(self.pred), k=self.n_instances)

    def fit(self, train_triples, valid_triples):
        self.create_model(train_triples, valid_triples)
        sz = (self.n_instances, self.n_instances, self.n_relations)

        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        x_s, x_o, x_p, Y = self.convert_data(train_triples, sampler=sampler)

        validation_x_s, validation_x_o, validation_p, validation_y = \
            self.convert_data(valid_triples, sampler=v_sampler)

        self.model.fit([x_s, x_o, x_p], Y,
                       batch_size=self.batch_size, epochs=self.training_epochs,
                       verbose=self.verbose, shuffle=True, callbacks=self.callbacks,
                       validation_data=(
                           [validation_x_s, validation_x_o, validation_p], validation_y))

    def batch_generator(self, train_triples, sampler):
        while 1:
            x_s, x_o, x_p, Y = self.convert_data(train_triples, sampler)
            idx = np.arange(Y.shape[0])
            np.random.shuffle(idx)
            for i in range(idx.shape[0] // self.batch_size):
                batch_idx = idx[i * self.batch_size:(i + 1) * self.batch_size]
                yield [x_s[batch_idx], x_o[batch_idx], x_p[batch_idx]], Y[batch_idx]

    def convert_data(self, triples, sampler=None):
        neg_triples = [sampler.sample(zip([t], [1])) for t in triples]

        x_s = [[triples[i][0]] + [nt[0][0] for nt in neg_triples[i]] for i in range(len(triples))]
        x_o = [[triples[i][1]] + [nt[0][1] for nt in neg_triples[i]] for i in range(len(triples))]
        x_p = [[triples[i][2]] + [nt[0][2] for nt in neg_triples[i]] for i in range(len(triples))]

        x_s = np.array(x_s, dtype="int32")
        x_o = np.array(x_o, dtype="int32")
        x_p = np.array(x_p, dtype="int32")

        y = np.hstack((np.ones((len(triples), 1)), np.zeros((len(triples), self.n_neg))))

        return x_s, x_o, x_p, y

    def fit_generator(self, train_triples, valid_triples):
        random.seed(88)
        self.create_model(train_triples, valid_triples)

        sz = (self.n_instances, self.n_instances, self.n_relations)
        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        validation_x_s, validation_x_o, validation_p, validation_y = \
            self.convert_data(valid_triples, sampler=v_sampler)

        self.model.fit_generator(self.batch_generator(train_triples, sampler),
                                 samples_per_epoch=len(train_triples) * (self.n_neg + 1),
                                 epochs=self.training_epochs, nb_worker=24, max_q_size=10, verbose=self.verbose,
                                 callbacks=self.callbacks,
                                 validation_data=(
                                     [validation_x_s, validation_x_o, validation_p], validation_y))


class MultilabelClassification(OptimizeRank):
    def joint_triple_score(self, s, o, p):
        subject_emb = self.ent_embeddings(s)
        object_emb = self.ent_embeddings(o)
        all_relations_emb = self.rel_embeddings(Lambda(lambda x: tf.range(0, self.n_relations))(s))
        return self.joint_triple_score_embs(subject_emb,object_emb,all_relations_emb)

    def create_model(self, train_triples=None, valid_triples=None):
        print("%d classes, %d instances, %d dim embeddings with hidden layers = %s" % (
            self.n_relations, self.n_instances, self.dimensions, str(self.layers)))

        self.input_s = input_s = Input(shape=(1,), dtype="int32", name="input_s")
        self.input_o = input_o = Input(shape=(1,), dtype="int32", name="input_o")
        self.input_p = input_p = Input(shape=(1,), dtype="int32", name="input_p")

        self.create_variables()

        predictions = self.joint_triple_score(input_s, input_o, input_p)

        self.model = Model(input=[self.input_s, self.input_o], output=predictions)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam())
        self.model.summary()

        self.pred = self.triple_score(input_s, input_o, input_p)

        _, self.rank = tf.nn.top_k(tf.transpose(predictions), k=self.n_instances)

    def convert_data(self, triples, sampler=None):
        data = defaultdict(lambda: np.zeros((self.n_relations,)))
        for s, o, p in triples:
            data[(s, o)][p] = 1
        if self.n_neg and sampler is not None:
            r_negs = sampler.sample(zip(triples, [1] * len(triples)))
            for (s, o, p), y in r_negs:
                data[(s, o)] = data[(s, o)]

        x_s = np.array([s for s, o in data.keys()], dtype="int32").reshape((-1, 1))
        x_o = np.array([o for s, o in data.keys()], dtype="int32").reshape((-1, 1))
        Y = np.array([y for y in data.values()])

        return x_s, x_o, Y

    def batch_generator(self, train_triples, sampler):
        while 1:
            x_s, x_o, Y = self.convert_data(train_triples, sampler)
            idx = np.arange(Y.shape[0])
            np.random.shuffle(idx)
            for i in range(idx.shape[0] // self.batch_size):
                batch_idx = idx[i * self.batch_size:(i + 1) * self.batch_size]
                yield [x_s[batch_idx], x_o[batch_idx]], Y[batch_idx]

    def fit_generator(self, train_triples, valid_triples):
        random.seed(88), np.random.seed(88)
        self.create_model(train_triples, valid_triples)

        sz = (self.n_instances, self.n_instances, self.n_relations)
        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        validation_x_s, validation_x_o, validation_y = self.convert_data(valid_triples, sampler=v_sampler)

        self.model.fit_generator(self.batch_generator(train_triples, sampler),
                                 samples_per_epoch=len(train_triples) * (self.n_neg + 1),
                                 epochs=self.training_epochs, nb_worker=1, max_q_size=10, verbose=self.verbose,
                                 callbacks=self.callbacks,
                                 validation_data=([validation_x_s, validation_x_o], validation_y))

    def fit(self, train_triples, valid_triples):

        random.seed(88), np.random.seed(88)
        self.create_model(train_triples, valid_triples)

        sz = (self.n_instances, self.n_instances, self.n_relations)
        all_triples = train_triples + valid_triples

        sampler = self.create_sampler(all_triples, sz)
        v_sampler = self.create_sampler(all_triples, sz)

        training_x_s, training_x_o, training_y = self.convert_data(train_triples, sampler=sampler)
        validation_x_s, validation_x_o, validation_y = self.convert_data(valid_triples, sampler=sampler)

        self.model.fit([training_x_s, training_x_o], training_y,
                       batch_size=self.batch_size, epochs=self.training_epochs,
                       verbose=self.verbose, shuffle=True, callbacks=self.callbacks,
                       validation_data=([validation_x_s, validation_x_o], validation_y))
