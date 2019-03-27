import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
import os
import json
from threading import Thread
from tqdm import tqdm
# import crash_on_ipy

MODELDIR = "../model"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.allow_soft_placement = True
config.log_device_placement = True

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

        with tf.device("/cpu:0"):
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

            cost = [tf.reduce_mean(tf.square(tf.subtract(c_, tf.concat([inputLayer[:, 0], inputLayer[:, 1]], 1))) / 2, axis=1)]

            for _ in range(2, options["max_seq_length"]):
                p = tf.nn.relu(tf.nn.bias_add(tf.einsum("jk,kl->jl", tf.concat([inputLayer[:, _], self.p[_-2]], 1), self.W_1), self.b_1))
                p = tf.div(p, tf.reshape(tf.norm(p, axis=1), (-1, 1)))
                c_ = tf.nn.relu(tf.nn.bias_add(tf.einsum("jk,kl->jl", p, self.W_2), self.b_2), name="c_")
                cost = tf.concat([cost, [tf.reduce_mean(tf.square(tf.subtract(c_, tf.concat([inputLayer[:, _], self.p[_-2]], 1))), axis=1)]], 0)
                # a = tf.concat([inputLayer[:, _], self.p[_-2]], 1)
                # b = tf.subtract(c_, a)
                # d = tf.reduce_mean(tf.square(b), axis=1)
                # import pdb;pdb.set_trace();
                # e = tf.concat([cost, [d]], 0)
                self.p = tf.concat([self.p, [p]], 0)

            all_cost = tf.reduce_sum(tf.transpose(cost, perm=[1, 0]) * mask, axis=1) / tf.reduce_sum(mask, axis=1, name="all_cost")
        # with tf.device("/gpu:0"):
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
            print("# epoch %s" % str(epoch))
            cost_epoch = 0
            train, test, train_mask, test_mask = train_test_split(data, m, shuffle=True)
            import math
            for batch in tqdm(range(int(math.ceil(len(train) / self.options["batch_size"])))):
                seqs = train[batch: (batch + 1) * self.options["batch_size"]]
                M = train_mask[batch: (batch + 1) * self.options["batch_size"]]

                X = self.seq2matrix(seqs)

                cost_batch, _ = self.sess.run([cost, optimizer], feed_dict={input: X, mask: M})
                cost_epoch += np.sum(cost_batch) / self.options["batch_size"]
            self.save()
            cost_epochs += cost_epoch
            if self.options['v'] == 2:
                print("#epoch {epoch} - cost: {cost_epoch}".format(epoch=epoch, cost_epoch=cost_epoch))
        import math
        print("{num_epochs} epochs - cost: {cost_epochs}".format(num_epochs=self.options["num_epochs"],
                                                                 cost_epochs=math.ceil(
                                                                     cost_epochs / self.options["num_epochs"])))


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
        sess = tf.Session(config=config)
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
    parse.add_argument("-n", dest="num_epochs", type=int, default=100)
    parse.add_argument("-b", dest="batch_size", type=int, default=100)
    parse.add_argument("-e", dest="embSize", type=int, default=100)
    parse.add_argument("-w", dest="W", type=int, default=100)
    parse.add_argument("-t", dest="datetime", type=str, default=None)
    parse.add_argument("--test", dest="test", action="store_true", default=False)
    parse.add_argument("--threshold", dest="threshold", help="rss memory threshold (G) [default: 100.0G]", type=float, default=100.0)
    parse.add_argument("--max-threshold", dest="max_threshold",help="max rss memory threshold updated to (G) [default: 200.0G]", type=float, default=200.0)
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

    stop = False
    threshold = args.threshold
    pid = os.getpid()

    def run(options):
        global stop
        model = RAE(options)
        model.train(seqs, mask)
        stop = True

    def mem_monitor(pid, max_threshold):
        import psutil
        global threshold
        global stop
        while not stop:
            p1 = psutil.Process(pid)
            mem = p1.memory_full_info()[0] / 1024 ** 3
            # print("\n" + "#" * 30)
            # print("mem: %.3fG" % (mem))
            # print("#" * 30)
            if mem > threshold:
                threshold *= 1.1
                if threshold > max_threshold:
                    threshold = max_threshold
                print('')
                print("# pid:", pid)
                print("# update threshold to %.3fG" % threshold)
                print("# need more %.3fG" % (threshold-mem))
                os.system("kill -19 %s" % pid)
            import time
            time.sleep(3)

    _task = Thread(target=run, args=[options], name="task")
    _mem = Thread(target=mem_monitor, args=[pid, args.max_threshold], name="memory_monitor")
    _task.start()
    _mem.start()
    _task.join()
    _mem.join()

    # model = RAE(options)
    # if not args.datetime:
    #     model.train(seqs, mask)
    # if args.test:
    #     # test = [[1, 2], [1, 2, 3], [1, 2, 3 ,4], [1, 2, 3, 4, 5], [4, 3, 2, 1], [4, 3, 2], [3, 2, 1]]
    #     test = [[1, 2, 3], [1, 2, 3, 4], [2, 3, 4], [3, 2, 1]]
    #     # test = [[0,0,0,0,0,0,0,0,0,0,0,0], [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7], [1,2,3,4,5,6,8,7], [1,2,3,4,5],[1,2,3,4],[1,2,3],[3,2,1],[4,2,1],[2,1]]
    #
    #     test_seqs = []
    #     test_seqs = test
    #
    #     test_mask = np.zeros((len(test), options["max_seq_length"] - 1))
    #     for k, _ in enumerate(test):
    #         test_seq = []
    #         for t, i in enumerate(_):
    #             # test_seq.append(types[i])
    #             try:
    #                 _t = t-1 if t-1 > 0 else 0
    #                 test_mask[k][t] = 1.
    #             except:
    #                 continue
    #         # test_seqs.append(test_seq)
    #
    #     # test_max_seq_len = max([len(_) for _ in test])
    #     result = model.transform(test_seqs, test_mask)
    #     for _ in range(len(test_seqs)):
    #         for i in range(_ + 1, len(test_seqs)):
    #             print(_, i, get_similarity(result[_], result[i]))

    # import pdb;pdb.set_trace();
