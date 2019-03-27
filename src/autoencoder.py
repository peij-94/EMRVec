import tensorflow as tf
from tensorflow.nn import relu

def encode_layer(input, params):
    with tf.name_scope("encode_layer"):
        hide_layer1 = relu(tf.add(tf.matmul(input, params["w"]["eh1"]), params["b"]["eh1"], name="h1"))
        hide_layer2 = relu(tf.add(tf.matmul(hide_layer1, params["w"]["eh2"]), params["b"]["eh2"], name="h2"))
        hide_layer3 = relu(tf.add(tf.matmul(hide_layer2, params["w"]["eh3"]), params["b"]["eh3"], name="h3"))

    return hide_layer3

def decode_layer(input, params):
    with tf.name_scope("decode_layer"):
        hide_layer1 = relu(tf.add(tf.matmul(input, params["w"]["dh1"]), params["b"]["dh1"], name="h1"))
        hide_layer2 = relu(tf.add(tf.matmul(hide_layer1, params["w"]["dh2"]), params["b"]["dh2"], name="h2"))
        hide_layer3 = relu(tf.add(tf.matmul(hide_layer2, params["w"]["dh3"]), params["b"]["dh3"], name="h3"))

    return hide_layer3

def input_layer(options):
    return tf.placeholder([None, options["num_inputs"]], dtype=tf.float32, name="inputs")

def init_params(options):
    initializer = tf.variance_scaling_initializer()
    weights = {
        "eh1": tf.Variable(initializer([options["num_inputs"], options["num_hide1"]]), dtype=tf.float32, name="W_eh1"),
        "eh2": tf.Variable(initializer([options["num_hide1"], options["num_hide2"]]), dtype=tf.float32, name="W_eh2"),
        "eh3": tf.Variable(initializer([options["num_hide2"], options["num_hide3"]]), dtype=tf.float32, name="W_eh3"),
        "dh1": tf.Variable(initializer([options["num_hide3"], options["num_hide2"]]), dtype=tf.float32, name="W_dh1"),
        "dh2": tf.Variable(initializer([options["num_hide2"], options["num_hide1"]]), dtype=tf.float32, name="W_dh2"),
        "dh3": tf.Variable(initializer([options["num_hide1"], options["num_inputs"]]), dtype=tf.float32, name="W_dh3"),
    }
    bias = {
        "eh1": tf.Variable(tf.zeros(options["num_hide1"]), dtype=tf.float32, name="b_eh1"),
        "eh2": tf.Variable(tf.zeros(options["num_hide2"]), dtype=tf.float32, name="b_eh2"),
        "eh3": tf.Variable(tf.zeros(options["num_hide3"]), dtype=tf.float32, name="b_eh3"),
        "dh1": tf.Variable(tf.zeros(options["num_hide2"]), dtype=tf.float32, name="b_dh1"),
        "dh2": tf.Variable(tf.zeros(options["num_hide1"]), dtype=tf.float32, name="b_dh2"),
        "dh3": tf.Variable(tf.zeros(options["num_inputs"]), dtype=tf.float32, name="b_dh3"),
    }

    return {"w": weights, "b": bias}

def build_model(options):

    # build model
    print("building model ...")
    params = init_params(options)
    inputs = input_layer(options)
    encoder = encode_layer(inputs, params)
    decoder = decode_layer(encoder, params)

    y_pred = decoder
    y_true = inputs

    loss = tf.reduce_mean((y_pred-y_true)/2)
    optimizer = tf.train.AdadeltaOptimizer().minimize(loss)

    return loss, optimizer




