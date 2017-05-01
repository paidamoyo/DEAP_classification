import tensorflow as tf


def mlp_neuron(layer_input, weights, biases, activation=True):
    mlp = tf.add(tf.matmul(layer_input, weights), biases)
    if activation:
        return tf.nn.relu(mlp)
    else:
        return mlp


# ### Helper-function for creating a new Fully-Connected Layer
def fc_layer(input,  # The previous layer.
             num_inputs,  # Num. inputs from prev. layer.
             num_outputs,  # Num. outputs.
             use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    layer_name = 'fully_connected'
    weights = create_weights(shape=[num_inputs, num_outputs], conv=False, name=layer_name)
    biases = create_biases(shape=[num_outputs], name=layer_name)
    layer = tf.matmul(input, weights) + biases
    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


def conv_layer(layer_name, input,  # The previous layer.
               num_input_channels,  # Num. channels in prev. layer.
               filter_size,  # Width and height of each filter.
               num_filters,  # Number of filters.
               use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            # Create new weights aka. filters with the given shape.
            weights = create_weights(shape=shape, conv=True, name=layer_name)
        with tf.name_scope('biases'):
            # Create new biases, one for each filter.
            biases = create_biases(shape=[num_filters], name=layer_name)
        with tf.name_scope('conv2d'):
            layer = tf.nn.conv2d(input=input,
                                 filter=weights,
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
            # A bias-value is added to each filter-channel.
            layer += biases
            tf.summary.histogram('pre_activations', layer)

            # Use pooling to down-sample the image resolution?
            if use_pooling:
                # This is 2x2 max-pooling, which means that we
                # consider 2x2 windows and select the largest value
                # in each window. Then we move 2 pixels to the next window.
                layer = tf.nn.max_pool(value=layer,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME')

        layer = tf.nn.relu(layer)
        # relu(max_pool(x)) == max_pool(relu(x)) we can
        tf.summary.histogram('activations', layer)

    return layer, weights


# ### Helper-function for flattening a layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    # layer_shape == [num_images, img_height, img_width, num_channels]
    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def normalized_mlp(layer_input, weights, biases, is_training, batch_norm):
    mlp = tf.add(tf.matmul(layer_input, weights), biases)
    if batch_norm:
        norm = batch_norm_wrapper(mlp, is_training)
        return tf.nn.relu(norm)
    else:
        return tf.nn.relu(mlp)


def create_nn_weights(layer, network, shape):
    h_vars = {}
    w_h = 'W_' + network + '_' + layer
    b_h = 'b_' + network + '_' + layer
    h_vars[w_h] = create_weights(shape=shape, name=w_h)
    h_vars[b_h] = create_biases([shape[1]], b_h)
    variable_summaries(h_vars[w_h], w_h)
    variable_summaries(h_vars[b_h], b_h)

    return h_vars[w_h], h_vars[b_h]


def create_biases(shape, name):
    print("name:{}, shape{}".format(name, shape))
    return tf.Variable(tf.constant(shape=shape, value=0.0), name=name)


def create_weights(shape, name, conv=False):
    print("name:{}, shape{}".format(name, shape))
    # initialize weights using Glorot and Bengio(2010) scheme
    if conv:
        a = tf.sqrt(6.0 / (shape[2] + shape[3]))
    else:
        a = tf.sqrt(6.0 / (shape[0] + shape[1]))
    # return tf.Variable(tf.random_normal(shape, stddev=tf.square(0.0001)), name=name)
    return tf.Variable(tf.random_uniform(shape, minval=-a, maxval=a, dtype=tf.float32))


def variable_summaries(var, summary_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def get_variables(name):
    var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)[0]
    return var


def one_label_tensor(label, num_ulab_batch, num_classes):
    indices = []
    values = []
    for i in range(num_ulab_batch):
        indices += [[i, label]]
        values += [1.]
    lab = tf.sparse_tensor_to_dense(
        tf.SparseTensor(indices=indices, values=values, dense_shape=[num_ulab_batch, num_classes]), 0.0)
    return lab


def batch_norm_wrapper(inputs, is_training):
    # http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    print("batch inputs {}".format(inputs.shape))

    offset = tf.Variable(tf.zeros([inputs.shape[1]]))
    scale = tf.Variable(tf.ones([inputs.shape[1]]))
    epsilon = 1e-4
    alpha = 0.999  # use numbers closer to 1 if you have more data

    def batch_norm():
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        print("batch mean {}, var {}".format(batch_mean.shape, batch_var.shape))
        train_mean = tf.assign(pop_mean,
                               pop_mean * alpha + batch_mean * (1 - alpha))
        train_var = tf.assign(pop_var,
                              pop_var * alpha + batch_var * (1 - alpha))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=offset, scale=scale,
                                             variance_epsilon=epsilon)

    def pop_norm():
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, offset=offset, scale=scale,
                                         variance_epsilon=epsilon)

    return tf.cond(is_training, batch_norm, pop_norm)


def dropout_normalised_mlp(layer_input, weights, biases, is_training, batch_norm, keep_prob=1):
    mlp = normalized_mlp(layer_input, weights, biases, is_training, batch_norm)  # apply DropOut to hidden layer
    drop_out = tf.nn.dropout(mlp, keep_prob)  # DROP-OUT here
    return drop_out


if __name__ == '__main__':
    y_ulab = one_label_tensor(2, 400, 10)
    with tf.Session() as session:
        y = session.run(y_ulab)
        print("y:{}, shape:{}".format(y, y.shape))
