import scipy.stats as stats
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
from keras.layers import merge
from keras.constraints import maxnorm



class L1NormFlat(Layer):
    def build(self, input_shape):
        assert len(input_shape) == 2

    def call(self, input, mask=None):
        return tf.reshape(tf.reduce_sum(input, 1), [-1, 1])

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

    def compute_output_shape_for(self, input_shape):
        return (input_shape[0], 1)

    def get_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


class L2NormFlat(L1NormFlat):
    def call(self, input, mask=None):
        input = tf.multiply(input, input)
        return tf.reshape(tf.sqrt(tf.reduce_sum(input, 1)), [-1, 1])


class L1Norm(Layer):
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.n_examples = input_shape[1]

    def call(self, input, mask=None):
        return tf.reduce_sum(input, 2)

    def get_output_shape_for(self, input_shape):
        return input_shape[:2]

    def compute_output_shape_for(self, input_shape):
        return input_shape[:2]

    def get_output_shape(self, input_shape):
        return input_shape[:2]

    def compute_output_shape(self, input_shape):
        return input_shape[:2]


class L2Norm(L1Norm):
    def call(self, input, mask=None):
        input = tf.multiply(input, input)
        return tf.sqrt(tf.reduce_sum(input, 2))


def max_pairwise_margin(y_true, y_pred, margin=0.5):
    y_pos = y_pred[:, 0]
    y_neg = y_pred[:, 1]
    return tf.reduce_sum(tf.maximum(0., y_pos - y_neg + margin))


def max_margin(y_true, y_pred):
    return tf.reduce_sum(tf.maximum(0., 1. - y_pred * y_true + y_pred * (1. - y_true)))



class Multiplication(Layer):
    def build(self, input_shape):
        print(input_shape)
        assert len(input_shape) == 2 and input_shape[0] == input_shape[1]

    def call(self, inputs, mask=None):
        a, b = inputs
        return tf.multiply(a, b)

    def get_output_shape(self, input_shape):
        return input_shape[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def compute_output_shape_for(self, input_shape):
        return input_shape[0]


class CircularCorrelation(Layer):
    def build(self, input_shape):
        assert len(input_shape) == 2 and input_shape[0] == input_shape[1] and len(input_shape[0])==3
        self.dim = input_shape[0][2]

    def call(self, inputs, mask=None):
        a, b = inputs
        b = tf.tile(b, multiples=[1, self.dim, 1])
        for i in range(1,self.dim):
            tf.manip.roll(b[i],shift=i, axis=2)
        print(b.get_shape())
        return tf.reshape(tf.matmul(a,b), shape=[-1,1,self.dim])

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def get_output_shape(self, input_shape):
        return input_shape[0]

    def compute_output_shape_for(self, input_shape):
        return input_shape[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class CircularCorrelationFT(CircularCorrelation):
    def call(self, inputs, mask=None):
        a, b = inputs
        a_fft = tf.fft(tf.complex(a, 0.0))
        b_fft = tf.fft(tf.complex(b, 0.0))
        ifft = tf.ifft(tf.conj(a_fft) * b_fft)
        return tf.cast(tf.real(ifft), 'float32')


class MultilabelDiff(Layer):
    def __init__(self, units):
        self.units = units

    def build(self, input_shape, W_constraint=None):
        assert len(input_shape) == 2 and input_shape[-1] == 1
        self.shape = input_shape
        self.shape[-1] = self.units
        self.W = self.add_weight(name="W", shape=self.shape, initializer='uniform', trainable=True, constraint=maxnorm(1, axis=1))
        self.trainable_weights = [self.W]

    def call(self, inputs, mask=None):
        assert type(inputs) is not list
        output = self.W - inputs
        return output

    def get_output_shape_for(self, input_shape):
        return self.shape

    def get_output_shape(self, input_shape):
        return self.shape

    def compute_output_shape_for(self, input_shape):
        return self.shape

    def compute_output_shape(self, input_shape):
        return self.shape

    def __getitem__(self, item):
        return self.W[:,item]

class ProjECombiner(Layer):
    def build(self, input_shape, W_constraint=None):
        self.shape = shape = input_shape[0]
        assert len(input_shape) == 2 and input_shape[0] == input_shape[1]
        self.W1 = self.add_weight(name="W1", shape=shape[1:], initializer='uniform', trainable=True, constraint=maxnorm(1, axis=1))
        self.W2 = self.add_weight(name="W2", shape=shape[1:], initializer='uniform', trainable=True, constraint=maxnorm(1, axis=1))
        self.b = self.add_weight(name="b", shape=shape[1:], initializer='uniform', trainable=True)
        self.trainable_weights = [self.W1, self.W2, self.b]

    def call(self, inputs, mask=None):
        assert len(inputs) == 2
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('SimpleCombinationOperator needs to be called on two tensors. Got: ' + str(inputs))
        output = inputs[0] * self.W1 + inputs[1] * self.W2 + self.b
        return output

    def get_output_shape_for(self, input_shape):
        return self.shape

    def get_output_shape(self, input_shape):
        return self.shape

    def compute_output_shape_for(self, input_shape):
        return self.shape

    def compute_output_shape(self, input_shape):
        return self.shape


class MatrixCombinationOperator(ProjECombiner):
    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        super(MatrixCombinationOperator, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert len(inputs) == self.n_inputs
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('SimpleCombinationOperator needs to be called on two tensors. Got: ' + str(inputs))

        xs = [merge([inputs[i], self.W[i]], mode="dot", dot_axes=(1, 2)) + self.b[i] for i in range(self.n_inputs)]

        output = xs[0]
        for i in range(1, self.n_inputs):
            output += xs[i]

        return output


class NeuralTensorLayer(Layer):
    def __init__(self, output_dim, input_dim=None, **kwargs):
        self.output_dim = output_dim  # k
        self.input_dim = input_dim  # d
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(NeuralTensorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        mean = 0.0
        std = 1.0
        # W : k*d*d
        k = self.output_dim
        d = self.input_dim
        initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k, d, d))
        initial_V_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2 * d, k))
        self.W = K.variable(initial_W_values)
        self.V = K.variable(initial_V_values)
        self.b = K.zeros((self.input_dim,))
        self.trainable_weights = [self.W, self.V, self.b]

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('BilinearTensorLayer must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        e1 = inputs[0]
        e2 = inputs[1]
        batch_size = K.shape(e1)[0]
        k = self.output_dim
        # print([e1,e2])
        feed_forward_product = K.dot(K.concatenate([e1, e2]), self.V)
        # print(feed_forward_product)
        bilinear_tensor_products = [K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1)]
        # print(bilinear_tensor_products)
        for i in range(k)[1:]:
            btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
            bilinear_tensor_products.append(btp)
        result = K.tanh(
            K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)
        # print(result)
        return result

    def get_output_shape_for(self, input_shape):
        # print (input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)

    def get_output_shape(self, input_shape):
        # print (input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)

    def compute_output_shape_for(self, input_shape):
        # print (input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)

    def compute_output_shape(self, input_shape):
        # print (input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)
