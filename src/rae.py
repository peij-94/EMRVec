import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
import os
import json
# import crash_on_ipy

MODELDIR = "../model"


class RAE:
    def __init__(self, options):
        self.options = options
        self.time = options.get("dateTime", datetime.strftime(datetime.now(), "%y%m%d_%H%M%S"))

        # model-params
        self.L = None
        self.W_1 = None
        self.W_2 = None
        self.b_1 = None
        self.b_2 = None
        # model-property
        self.p = None
        self.modelname = None
        self.sess = None

        self.init_params()
        self.trained = False

    def init_params(self):

        self.modelname = "rae_%s.model" % (self.time)

        initializer = tf.variance_scaling_initializer()
        self.L = tf.Variable(np.random.normal(size=self.options["inputSize"] * self.options["embSize"]).reshape(
            self.options["inputSize"], self.options["embSize"]), dtype=tf.float32, name="L")
        self.W_1 = tf.Variable(initializer([self.options["embSize"] * 2, self.options["W"]]), dtype=tf.float32,
                               name="W_1")
        self.W_2 = tf.Variable(initializer([self.options["W"], self.options["embSize"] * 2]), dtype=tf.float32,
                               name="W_2")
        self.b_1 = tf.Variable(initializer([self.options["W"]]), dtype=tf.float32, name="b_1")
        self.b_2 = tf.Variable(initializer([self.options["embSize"] * 2]), dtype=tf.float32, name="b_2")

        self.sess = self.getSession()
        if self.options.get("dateTime", None):
            self.L = self.sess.run("L:0")
            self.W_1 = self.sess.run("W_1:0")
            self.W_2 = self.sess.run("W_2:0")
            self.b_1 = self.sess.run("b_1:0")
            self.b_2 = self.sess.run("b_2:0")


    def build_model(self, options):

        input = tf.placeholder(tf.float32, [None, options["max_seq_length"], options["inputSize"]], name="input")
        # input = tf.transpose(input, perm=[1, 0, 2])
        inputLayer = tf.einsum("jkl,lm->jkm", input, self.L, name="inputLayer")
        # inputLayer = tf.matmul(input, self.L)
        mask = tf.placeholder(tf.float32, [None, options["max_seq_length"] - 1], name="mask")
        # mask = tf.transpose(mask, perm=[1, 0, 2])

        p = tf.nn.relu(
            tf.nn.bias_add(tf.einsum("jl,lm->jm", tf.concat([inputLayer[:, 0], inputLayer[:, 1]], 1), self.W_1),
                           self.b_1), name="p")
        p = tf.div(p, tf.reshape(tf.norm(p, axis=1), (-1, 1)))

        self.p = [p]

        c_ = tf.nn.relu(tf.nn.bias_add(tf.einsum("jl,lm->jm", p, self.W_2), self.b_2), name="c_")
        c1 = c_[:, :options["embSize"]]
        c2 = c_[:, options["embSize"]:]

        cost = [tf.add(tf.reduce_mean(tf.square(tf.subtract(c1, inputLayer[:, 0])) / 2, axis=1),
                       tf.reduce_mean(tf.square(tf.subtract(c2, inputLayer[:, 1])) / 2, axis=1))]

        for _ in range(2, options["max_seq_length"]):
            c = tf.concat([inputLayer[:, _], p], 1)
            p_ = tf.nn.relu(tf.nn.bias_add(tf.einsum("jk,kl->jl", c, self.W_1), self.b_1), name="p_")
            c_ = tf.nn.relu(tf.nn.bias_add(tf.einsum("jk,kl->jl", p_, self.W_2), self.b_2), name="c_")
            c1 = c_[:, :options["embSize"]]
            c2 = c_[:, options["embSize"]:]
            cost_ = tf.add((1 / (1 + _) * tf.reduce_mean(tf.square(tf.subtract(c1, inputLayer[:, _])), axis=1)),
                           (_ / (1 + _) * tf.reduce_mean(tf.square(tf.subtract(c2, p)), axis=1)), name="cost_")
            p = p_
            cost = tf.concat([cost, [cost_]], 0)
            self.p = tf.concat([self.p, [p]], 0)

        cost = tf.transpose(cost, perm=[1, 0], name="cost")
        all_cost = tf.reduce_sum(cost * mask, axis=1) / tf.reduce_sum(mask, axis=1, name="all_cost")
        optimizer = tf.train.AdadeltaOptimizer().minimize(all_cost)

        return input, mask, all_cost, optimizer

    def train(self, data, m):
        self.trained = True
        input, mask, cost, optimizer = self.build_model(self.options)
        init = tf.global_variables_initializer()

        self.sess.run(init)
        # sess.graph.finalize()
        cost_epochs = 0
        for epoch in range(self.options["num_epochs"]):
            cost_epoch = 0
            train, test, train_mask, test_mask = train_test_split(data, m, shuffle=True)
            import math
            for batch in range(int(math.ceil(len(train) / self.options["batch_size"]))):
                seqs = train[batch: (batch + 1) * self.options["batch_size"]]
                M = train_mask[batch: (batch + 1) * self.options["batch_size"]]

                X = self.seq2matrix(seqs)

                cost_batch, _ = self.sess.run([cost, optimizer], feed_dict={input: X, mask: M})
                cost_epoch += np.sum(cost_batch) / self.options["batch_size"]

            cost_epochs += cost_epoch
            if self.options['v'] == 2:
                print("#epoch {epoch} - cost: {cost_epoch}".format(epoch=epoch, cost_epoch=cost_epoch))
        import math
        print("{num_epochs} epochs - cost: {cost_epochs}".format(num_epochs=self.options["num_epochs"],
                                                                 cost_epochs=math.ceil(
                                                                     cost_epochs / self.options["num_epochs"])))
        self.save()

    def seq2matrix(self, seqs):
        X = np.zeros((len(seqs), self.options["max_seq_length"], self.options["inputSize"]), dtype=np.float32)
        for i, seq in enumerate(seqs):
            for j, item in enumerate(reversed(seq)):
                X[i][j][item] = 1.
        return X

    def transform(self, data, mask):
        # TODO
        input = "input:0"
        if not self.trained:
            input = self.build_model(self.options)[0]
            init = tf.global_variables_initializer()
            self.sess.run(init)
            input = input

        X = self.seq2matrix(data)
        return [self.sess.run([self.p], feed_dict={input: X, "mask:0": mask})[0][i][_] for _, i in enumerate((mask.sum(1)-2).astype(int).tolist())]

    def save(self):
        saver = tf.train.Saver()
        mdir = "%s/%s" % (MODELDIR, self.modelname)
        os.makedirs(mdir, exist_ok=True)
        saver.save(self.sess, os.path.join(mdir, self.modelname))
        json.dump(self.options, open(os.path.join(mdir, "rae_%s.params" % self.time), "w"))

    def getSession(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        try:
            saver.restore(sess, self.modelname)
        except:
            pass
        finally:
            return sess


def get_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("-n", dest="num_epochs", type=int, default=0)
    parse.add_argument("-b", dest="batch_size", type=int, default=0)
    parse.add_argument("-e", dest="embSize", type=int, default=0)
    parse.add_argument("-w", dest="W", type=int, default=0)
    parse.add_argument("-t", dest="datetime", type=str, default=None)
    parse.add_argument("--test", dest="test", action="store_true", default=False)
    args = parse.parse_args()

    import pickle
    if not args.test:
        seqs = pickle.load(open("../data/diagnoses_mimiciii.pkl", "rb"))
        types = pickle.load(open("../data/diagnoses_mimiciii.types", "rb"))
        max_seq_len = max([len(_) for _ in seqs])
        mask = np.zeros((len(seqs), max_seq_len - 1))

        for k, _ in enumerate(seqs):
            for t, i in enumerate(_):
                try:
                    _t = t-1 if t-1 > 0 else t
                    mask[k][_t] = 1.
                except:
                    continue

    else:
        data = [[1, 2, 3, 4, 5], [3, 5, 6], [1, 6, 3, 2, 4], [2, 2, 3, 3], [1, 2], [3, 1]]
        seqs = []
        types = {}
        max_seq_len = max([len(_) for _ in data])
        mask = np.zeros((len(data), max_seq_len - 1))

        for k, _ in enumerate(data):
            seq = []
            for t, i in enumerate(_):
                if i not in types:
                    types[i] = len(types)
                seq.append(types[i])
                try:
                    mask[k][t] = 1.
                except:
                    continue
            seqs.append(seq)

    if args.datetime:
        options = json.load(
            open(os.path.join(MODELDIR, "rae_%s.model" % args.datetime, "rae_%s.params" % args.datetime), "r"))
        options["datetime"] = args.datetime
    else:
        options = {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "embSize": args.embSize,
            "inputSize": len(types),
            "W": args.W,
            "max_seq_length": max_seq_len,
            'v': 2
        }

    model = RAE(options)
    if not args.datetime:
        model.train(seqs, mask)
    if args.test:
        # test = [[1, 2], [1, 2, 3], [1, 2, 3 ,4], [1, 2, 3, 4, 5], [4, 3, 2, 1], [4, 3, 2], [3, 2, 1]]
        test = [[1, 2, 3], [1, 2, 3, 4], [2, 3, 4], [3, 2, 1]]
        # test = [[0,0,0,0,0,0,0,0,0,0,0,0], [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7], [1,2,3,4,5,6,8,7], [1,2,3,4,5],[1,2,3,4],[1,2,3],[3,2,1],[4,2,1],[2,1]]

        test_seqs = []
        test_seqs = test

        test_mask = np.zeros((len(test), options["max_seq_length"] - 1))
        for k, _ in enumerate(test):
            test_seq = []
            for t, i in enumerate(_):
                # test_seq.append(types[i])
                try:
                    _t = t-1 if t-1 > 0 else 0
                    test_mask[k][t] = 1.
                except:
                    continue
            # test_seqs.append(test_seq)

        # test_max_seq_len = max([len(_) for _ in test])
        result = model.transform(test_seqs, test_mask)
        for _ in range(len(test_seqs)):
            for i in range(_ + 1, len(test_seqs)):
                print(_, i, get_similarity(result[_], result[i]))

    # import pdb;pdb.set_trace();
