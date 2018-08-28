from training import PairwiseMargin, OptimizeRank, MultilabelClassification
from keras.layers import Dense, Dropout, Embedding, Activation, Merge, Input, merge, Flatten, Lambda, \
    Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Conv3D, MaxPooling3D, Reshape, LocallyConnected1D, LocallyConnected2D, \
    AveragePooling2D, Multiply, Dot, Concatenate
from keras_layers import CircularCorrelationFT, CircularCorrelation, ProjECombiner, L2NormFlat, L2Norm, L1NormFlat, L1Norm
import math
from keras import backend as K
import tensorflow as tf


class TransE_PairwiseMargin(PairwiseMargin):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        diff = Lambda(lambda x: x[0] + x[2] - x[1])([subject_emb, object_emb, relation_emb])
        diff = Flatten()(diff)
        score = L2NormFlat()(diff)
        score = Lambda(lambda x: -x)(score)
        score = self.sigm_act(score)
        return score


class HolE_PairwiseMargin(PairwiseMargin):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        so = CircularCorrelationFT()([subject_emb, object_emb])
        score = Dot([2,2])([so, relation_emb])
        score = Flatten()(score)
        score = self.sigm_act(score)
        return score


class DistMult_PairwiseMargin(PairwiseMargin):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        diff = Multiply()([subject_emb, object_emb, relation_emb])
        score = L1Norm()(diff)
        score = self.sigm_act(score)
        return score


class NTN_Pairwise(PairwiseMargin):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        raise NotImplementedError()


class ComplexE_PairwiseMargin(PairwiseMargin):
    def create_variables(self):
        self.ent_embeddings_real = Embedding(input_dim=self.n_instances, output_dim=self.dimensions, trainable=True,
                                             W_constraint=self.constraint)
        self.ent_embeddings_imag = Embedding(input_dim=self.n_instances, output_dim=self.dimensions, trainable=True,
                                             W_constraint=self.constraint)
        self.rel_embeddings_real = Embedding(input_dim=self.n_relations, output_dim=self.rel_dimensions, trainable=True,
                                             W_constraint=self.constraint)
        self.rel_embeddings_imag = Embedding(input_dim=self.n_relations, output_dim=self.rel_dimensions, trainable=True,
                                             W_constraint=self.constraint)

    def triple_score(self, s, o, p):
        subject_emb = [self.ent_embeddings_real(s), self.ent_embeddings_imag(s)]
        object_emb = [self.ent_embeddings_real(o), self.ent_embeddings_imag(o)]
        relation_emb = [self.rel_embeddings_real(p), self.rel_embeddings_imag(p)]
        return self.triple_score_embs(subject_emb, object_emb, relation_emb)

    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        s_real, s_imag = subject_emb
        o_real, o_imag = object_emb
        p_real, p_imag = relation_emb
        s1 = Multiply()([s_real, o_real, p_real])
        s2 = Multiply()([s_imag, o_imag, p_real])
        s3 = Multiply()([s_real, o_imag, p_imag])
        s4 = Multiply()([s_imag, o_real, p_imag])
        s1 = L1Norm()(s1)
        s2 = L1Norm()(s2)
        s3 = L1Norm()(s3)
        s4 = L1Norm()(s4)
        score = Lambda(lambda s: s[0] + s[1] + s[2] - s[3])([s1, s2, s3, s4])
        score = self.sigm_act(score)
        return score


class ConvEDettmers_PairwiseMargin(PairwiseMargin):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        raise NotImplementedError()


class TransR_PairwiseMargin(PairwiseMargin):
    def __init__(self, n_rel_dim=10, *args, **kwargs):
        super(TransR_PairwiseMargin, self).__init__(*args, **kwargs)
        self.rel_dimensions = n_rel_dim

    def create_variables(self):
        super(TransR_PairwiseMargin, self).create_variables()
        self.projection = Embedding(input_dim=self.n_relations, output_dim=self.dimensions * self.rel_dimensions,
                                    trainable=True,
                                    W_constraint=self.constraint)

    def triple_score(self, s, o, p):
        subject_emb = self.ent_embeddings(s)
        object_emb = self.ent_embeddings(o)
        relation_emb = self.rel_embeddings(p)
        relation_proj = self.projection(p)
        return self.triple_score_embs(subject_emb, object_emb, relation_emb, relation_proj)

    def triple_score_embs(self, subject_emb, object_emb, relation_emb, relation_proj):
        relation_proj = Reshape((self.dimensions, self.rel_dimensions))(relation_proj)
        subject_emb = Reshape((1, self.dimensions))(subject_emb)
        object_emb = Reshape((1, self.dimensions))(object_emb)
        subject_emb = Dot(axes=[2, 1])([subject_emb, relation_proj])
        object_emb = Dot(axes=[2, 1])([object_emb, relation_proj])
        subject_emb = Flatten()(subject_emb)
        object_emb = Flatten()(object_emb)
        relation_emb = Flatten()(relation_emb)
        diff = Lambda(lambda x: x[0] + x[2] - x[1])([subject_emb, object_emb, relation_emb])
        return L2NormFlat()(diff)


class ProjE_PairwiseMargin(PairwiseMargin):
    def create_variables(self):
        super(ProjE_PairwiseMargin, self).create_variables()
        self.sp_combiner = ProjECombiner()
        self.op_combiner = ProjECombiner()

    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        sp = self.sp_combiner([subject_emb, relation_emb])
        op = self.op_combiner([object_emb, relation_emb])
        spo = Dot(axes=[2, 2])([sp, object_emb])
        ops = Dot(axes=[2, 2])([op, subject_emb])
        score = Lambda(lambda x: - (x[0] + x[1]))([spo, ops])
        score = Flatten()(score)
        score = self.sigm_act(score)
        return score


class ConvE_PairwiseMargin(PairwiseMargin):
    def __init__(self, n_filters=10, *args, **kwargs):
        super(ConvE_PairwiseMargin, self).__init__(*args, **kwargs)
        self.n_filters = n_filters

    def create_variables(self):
        self.filter_dim = 3
        self.kernel_size = (self.filter_dim, self.filter_dim)
        self.stride = 1
        self.max_pool_dim = 2
        self.emb_side_dim = int(self.dimensions ** 0.5)
        self.dimensions = self.emb_side_dim ** 2
        self.n_channels = 1
        self.emb_shape = (self.emb_side_dim, self.emb_side_dim, self.n_channels)
        self.rel_dimensions = (int(math.ceil((self.emb_side_dim - 2) / self.max_pool_dim))) ** 2 * self.n_filters

        super(ConvE_PairwiseMargin, self).create_variables()

        self.shared_conv = Conv2D(self.n_filters, kernel_size=self.kernel_size, subsample=(self.stride, self.stride),
                                  padding=self.border_mode, activation=self.activation, W_constraint=self.constraint)
        self.shared_maxpool = MaxPooling2D((self.max_pool_dim, self.max_pool_dim), padding=self.border_mode)
        self.shared_act = Activation("relu")

    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        subject_emb = Reshape(self.emb_shape, input_shape=(self.dimensions * self.n_channels,))(subject_emb)
        object_emb = Reshape(self.emb_shape, input_shape=(self.dimensions * self.n_channels,))(object_emb)

        x = Lambda(lambda x: x[0] - x[1])([subject_emb, object_emb])
        x = self.shared_conv(x)
        x = self.shared_maxpool(x)
        x = self.shared_act(x)

        x = Flatten()(x)
        relation_emb = Flatten()(relation_emb)
        diff = Lambda(lambda x: x[0] - x[1])([x, relation_emb])
        score = L2NormFlat()(diff)

        score = self.sigm_act(score)
        return score


class TransE_OptimizeRank(OptimizeRank):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        diff = Lambda(lambda x: x[0] + x[2] - x[1])([subject_emb, object_emb, relation_emb])
        score = L2Norm()(diff)
        score = Lambda(lambda x: -x)(score)
        score = self.sigm_act(score)
        return self.sigm_act(score)

    def joint_triple_score_embs(self, subject_emb, object_emb, relation_emb):
        return self.triple_score_embs(subject_emb, object_emb, relation_emb)


class DistMult_OptimizeRank(TransE_OptimizeRank):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        diff = Multiply()([subject_emb, object_emb, relation_emb])
        score = L1Norm()(diff)
        score = self.sigm_act(score)
        return score


class HolE_OptimizeRank(DistMult_OptimizeRank):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        so = CircularCorrelationFT()([subject_emb, object_emb])
        score = Multiply()([so, relation_emb])
        score = Lambda(lambda x: tf.reduce_sum(score, axis=2))(score)
        score = self.sigm_act(score)
        return Lambda(lambda x: -x)(score)


class ComplexE_OptimizeRank(OptimizeRank):
    def create_variables(self):
        self.ent_embeddings_real = Embedding(input_dim=self.n_instances, output_dim=self.dimensions, trainable=True,
                                             W_constraint=self.constraint)
        self.ent_embeddings_imag = Embedding(input_dim=self.n_instances, output_dim=self.dimensions, trainable=True,
                                             W_constraint=self.constraint)
        self.rel_embeddings_real = Embedding(input_dim=self.n_relations, output_dim=self.rel_dimensions, trainable=True,
                                             W_constraint=self.constraint)
        self.rel_embeddings_imag = Embedding(input_dim=self.n_relations, output_dim=self.rel_dimensions, trainable=True,
                                             W_constraint=self.constraint)

    def triple_score(self, s, o, p):
        subject_emb = [self.ent_embeddings_real(s), self.ent_embeddings_imag(s)]
        object_emb = [self.ent_embeddings_real(o), self.ent_embeddings_imag(o)]
        relation_emb = [self.rel_embeddings_real(p), self.rel_embeddings_imag(p)]
        return self.triple_score_embs(subject_emb, object_emb, relation_emb)

    def joint_triple_score(self, s, o, p):
        return self.triple_score(s, o, p)

    def joint_triple_score_embs(self, subject_emb, object_emb, relation_emb):
        return self.triple_score_embs(self, subject_emb, object_emb, relation_emb)

    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        s_real, s_imag = subject_emb
        o_real, o_imag = object_emb
        p_real, p_imag = relation_emb
        s1 = Multiply()([s_real, o_real, p_real])
        s2 = Multiply()([s_imag, o_imag, p_real])
        s3 = Multiply()([s_real, o_imag, p_imag])
        s4 = Multiply()([s_imag, o_real, p_imag])
        s1 = L1Norm()(s1)
        s2 = L1Norm()(s2)
        s3 = L1Norm()(s3)
        s4 = L1Norm()(s4)
        score = Lambda(lambda s: s[0] + s[1] + s[2] - s[3])([s1, s2, s3, s4])
        score = self.sigm_act(score)
        return score


class ConvE_OptimizeRank(OptimizeRank):
    def __init__(self, n_filters=10, *args, **kwargs):
        super(ConvE_OptimizeRank, self).__init__(*args, **kwargs)
        self.n_filters = n_filters

    def create_variables(self):
        self.filter_dim = 3
        self.kernel_size = (self.filter_dim, self.filter_dim)
        self.stride = 1
        self.max_pool_dim = 2
        self.emb_side_dim = int(self.dimensions ** 0.5)
        self.dimensions = self.emb_side_dim ** 2
        self.n_channels = 1
        self.emb_shape = (self.emb_side_dim, self.emb_side_dim, self.n_channels)
        self.rank_emb_shape = ((self.n_neg + 1), self.emb_side_dim, self.emb_side_dim, self.n_channels)
        self.rel_dimensions = (int(math.ceil((self.emb_side_dim - 2) / self.max_pool_dim))) ** 2 * self.n_filters

        super(ConvE_OptimizeRank, self).create_variables()

        self.shared_conv = Conv2D(self.n_filters, kernel_size=self.kernel_size, subsample=(self.stride, self.stride),
                                  padding=self.border_mode, activation=self.activation, W_constraint=self.constraint)
        self.shared_maxpool = MaxPooling2D((self.max_pool_dim, self.max_pool_dim), padding=self.border_mode)
        self.relu_act = Activation("relu")
        self.sigm_act = Activation("sigmoid")
        self.subtract = Lambda(lambda x: x[0] - x[1])
        self.reshape_emb = Reshape(self.emb_shape)
        self.flatten = Flatten()
        self.l2norm = L2NormFlat()

    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        subject_emb = self.reshape_emb(subject_emb)
        object_emb = self.reshape_emb(object_emb)
        x = self.subtract([subject_emb, object_emb])
        x = self.shared_conv(x)
        x = self.shared_maxpool(x)
        x = self.relu_act(x)

        x = self.flatten(x)
        # relation_emb = Flatten()(relation_emb)
        diff = self.subtract([x, relation_emb])
        score = self.l2norm(diff)

        score = self.sigm_act(score)
        return score

    def joint_triple_score_embs(self, subject_emb, object_emb, relation_emb):
        subject_emb = Reshape(self.rank_emb_shape, input_shape=(self.n_neg + 1, self.dimensions * self.n_channels,))(
            subject_emb)
        object_emb = Reshape(self.rank_emb_shape, input_shape=(self.n_neg + 1, self.dimensions * self.n_channels,))(
            object_emb)
        x = Lambda(lambda x: x[0] - x[1])([subject_emb, object_emb])

        scores = []
        for i in range(0, self.n_neg + 1):
            column_i = Lambda(lambda x: x[:, i])
            x_i = column_i(x)
            x_i = self.shared_conv(x_i)
            x_i = self.shared_maxpool(x_i)
            x_i = self.relu_act(x_i)
            x_i = self.flatten(x_i)
            r_i = column_i(relation_emb)
            diff = self.subtract([x_i, r_i])
            score = self.l2norm(diff)
            scores.append(self.sigm_act(score))
        return Concatenate()(scores)


class ConvE_Multilabel(MultilabelClassification):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        diff = Lambda(lambda x: x[0] + x[2] - x[1])([subject_emb, object_emb, relation_emb])
        diff = Flatten()(diff)
        score = L2NormFlat()(diff)
        return self.sigm_act(score)

    def joint_triple_score_embs(self, subject_emb, object_emb, relation_emb):
        print(subject_emb.get_shape(), object_emb.get_shape(), relation_emb.get_shape())
        diff = Lambda(lambda x: x[0] + x[2] - x[1])([subject_emb, object_emb, relation_emb])
        score = L2Norm()(diff)
        return self.sigm_act(score)


class TransE_Multilabel(MultilabelClassification):
    def triple_score_embs(self, subject_emb, object_emb, relation_emb):
        diff = Lambda(lambda x: x[0] + x[2] - x[1])([subject_emb, object_emb, relation_emb])
        diff = Flatten()(diff)
        score = L2NormFlat()(diff)
        return self.sigm_act(score)

    def joint_triple_score_embs(self, subject_emb, object_emb, relation_emb):
        print(subject_emb.get_shape(), object_emb.get_shape(), relation_emb.get_shape())
        diff = Lambda(
            lambda x: tf.tile(x[0], multiples=[self.n_relations, 1]) - tf.tile(x[1], multiples=[self.n_relations, 1]) +
                      x[2])([subject_emb, object_emb, relation_emb])
        score = L2Norm()(diff)
        return self.sigm_act(score)
