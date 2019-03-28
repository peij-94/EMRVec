import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from datetime import datetime
import numpy as np
import os
import json
import time
import argparse
from threading import Thread
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from tensorboard.main import run_main
# import crash_on_ipy

MODELDIR = "../model"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

def get_seqs_input(data):
    e_lens = []
    d_lens = []
    e_data = []
    d_data_in = []
    d_data_out = []
    max_len = max([len(_) for _ in data])
    for _arr in data:
        e_lens.append(len(_arr))
        d_lens.append(len(_arr) + 1)
        e_data.append(np.pad(_arr, (0, max_len - len(_arr)), 'constant').tolist())
        d_data_tmp = np.pad(_arr, (0, max_len - len(_arr) + 1), 'constant')
        d_data_in.append(d_data_tmp.tolist())
        d_data_out.append(np.pad(d_data_tmp[:-1], (1, 0), 'constant').tolist())
    return e_data, e_lens, d_data_in, d_data_out, d_lens

class AttentionModel:

    def __init__(self, options):

        self.options = options
        self.EOS = 0
        self.GO = 0
        self.is_summaries = self.options["is_summaries"]
        self.graph = tf.get_default_graph()

        self.time = options.get("dateTime", datetime.strftime(datetime.now(), "%y%m%d_%H%M%S"))
        self.modelname = "attn_%s.model" % (self.time)
        self.sess = self.getSession()
        self.trained = False

        self.FWpath = os.path.join('..', 'summaries')


    def getSession(self):
        # init session
        sess = tf.Session()
        try:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.join(MODELDIR, self.modelname, self.modelname))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            pass
        finally:
            return sess

    def build_model(self):
        with self.graph.as_default():
            self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name='encoder_inputs')
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name='decoder_inputs')
            self.decoder_targets = tf.placeholder(tf.int32, shape=[None, None], name='decoder_targets')
            self.encoder_length = tf.placeholder(tf.int32, shape=[None], name='encoder_length')
            self.decoder_length = tf.placeholder(tf.int32, shape=[None], name='decoder_length')

            with tf.variable_scope("embedding"):
                encoder_embedding = tf.Variable(tf.truncated_normal(shape=[self.options["vocab_size"], self.options["embedding_dim"]], name='encoder_embedding'))
                decoder_embedding = tf.Variable(tf.truncated_normal(shape=[self.options["vocab_size"], self.options["embedding_dim"]], name='decoder_embedding'))

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.encoder_inputs)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, self.decoder_inputs)

            with tf.name_scope("encoder"):
                # Multiple-layers LSTM Layer as Encoder
                encoder_layers = [tf.nn.rnn_cell.BasicLSTMCell(self.options["cell_size"]) for _ in range(2)]
                encoder = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)
                # encoder output
                encoder_all_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder, self.encoder_inputs_embedded, sequence_length=self.encoder_length, dtype=tf.float32, time_major=False)
                if self.is_summaries:
                    tf.summary.histogram('encoder_all_outputs', encoder_all_outputs)

            with tf.name_scope("decoder"):
                # Multiple-layers LSTM Layer as Decoder
                decoder_layers = [tf.nn.rnn_cell.BasicLSTMCell(self.options["cell_size"]) for _ in range(2)]
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)
                # define attention algorithm
                attention_mechanism = LuongAttention(num_units=self.options["cell_size"], memory=encoder_all_outputs, memory_sequence_length=self.encoder_length)
                # connection of decoder & attention
                attention_decoder = AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism, alignment_history=True, output_attention=True)
                # init the starting state of attention
                attention_initial_state = attention_decoder.zero_state(tf.shape(self.encoder_inputs)[0], tf.float32).clone(cell_state=encoder_final_state)
                # define decoder input
                train_helper = TrainingHelper(self.decoder_inputs_embedded, self.decoder_length)
                fc_layer = tf.layers.Dense(self.options["vocab_size"])
                # define decoder
                train_decoder = BasicDecoder(cell=attention_decoder, helper=train_helper, initial_state=attention_initial_state, output_layer=fc_layer)
                # decoder output
                logits, final_state, final_sequence_lengths = dynamic_decode(train_decoder)
                decoder_logits = logits.rnn_output
                # attention matric
                self.train_attention_matrices = final_state.alignment_history.stack(name="train_attention_matrix")
                if self.is_summaries:
                    tf.summary.histogram('decoder_logits', decoder_logits)
                    tf.summary.histogram('train_attention_matrix', self.train_attention_matrices)


            with tf.name_scope("train"):
                maxlen = tf.reduce_max(self.decoder_length, name="mask_max_len")
                mask = tf.sequence_mask(self.decoder_length, maxlen=maxlen, dtype=tf.float32, name="mask")
                decoder_labels = tf.one_hot(self.decoder_targets, depth=self.options["vocab_size"], dtype=tf.int32, name="decoder_labels")
                stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=decoder_labels, logits=decoder_logits, name="cross_entropy")
                # loss
                _loss = tf.multiply(stepwise_cross_entropy, mask)
                self.loss = tf.reduce_sum(_loss, name="loss")
                optimizer = tf.train.AdadeltaOptimizer()
                self.train_op = optimizer.minimize(self.loss)
                if self.is_summaries:
                    tf.summary.histogram('loss', self.loss)

            with tf.name_scope("inference"):
                start_tokens = tf.tile([self.GO], [self.options["batch_size"]])
                inference_helper = GreedyEmbeddingHelper(embedding=decoder_embedding, start_tokens=start_tokens, end_token=self.EOS)
                inference_decoder = BasicDecoder(cell=attention_decoder, helper=inference_helper, initial_state=attention_initial_state, output_layer=fc_layer)
                inference_ouputs, final_inference_state, _ = dynamic_decode(inference_decoder)
                self.inference_attention_matrices = final_inference_state.alignment_history.stack(name="inference_attention_matrix")

            self.merged = tf.summary.merge_all()

    def rnnCell(self):
        # rnn cell(s)
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(options["cell_size"])
        if options.get("GRU", False):
            single_cell = tf.nn.rnn_cell.GRUCell(options["cell_size"])
        cell = single_cell
        if options["num_layers"] > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(options["num_layers"])])
        return cell

    def train(self, data):
        with self.graph.as_default():
            print("# start building model")
            self.build_model()
            self.sess.run(tf.global_variables_initializer())

            fwriter = tf.summary.FileWriter(self.FWpath, self.sess.graph)

            print("# finish building model")
            print("# start training")
            # self.sess.graph.finalize()
            loss_epochs = 0
            for e in range(self.options["num_epochs"]):
                print("# epoch %d" % e)
                loss_epoch = 0
                inds = np.arange(data.shape[0])
                np.random.shuffle(inds)
                import math
                for batch in tqdm(range(int(math.ceil(data.shape[0] / self.options["batch_size"])))):
                    seqs = data[inds[batch * self.options["batch_size"]: (batch + 1) * self.options["batch_size"]]]
                    e_data, e_len, d_data_in, d_data_out, d_len = get_seqs_input(seqs)
                    feed = {
                        self.encoder_inputs: e_data,
                        self.decoder_inputs: d_data_in,
                        self.decoder_targets: d_data_out,
                        self.encoder_length: e_len,
                        self.decoder_length: d_len
                    }
                    if self.is_summaries:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        attn_, loss_, merged_,  _ = self.sess.run([self.train_attention_matrices, self.loss, self.merged, self.train_op], feed_dict=feed, run_metadata=run_metadata, options=run_options)
                        if batch % 10 == 0:
                            fwriter.add_run_metadata(run_metadata, '# step %s' % str(e*self.options["batch_size"]+batch))
                            fwriter.add_summary(merged_, e*self.options["batch_size"]+batch)
                    else:
                        attn_, loss_,  _ = self.sess.run([self.train_attention_matrices, self.loss, self.train_op], feed_dict=feed)
                    loss_epoch += np.sum(loss_) / self.options["batch_size"]
                loss_epochs += loss_epoch

                self.save(global_step=self.options["num_epochs"])

                if self.options['v'] == 2:
                    print("#epoch {epoch} - cost: {cost_epoch}".format(epoch=e+1, cost_epoch=loss_epoch))
            import math
            print("{num_epochs} epochs - cost: {cost_epochs}".format(num_epochs=self.options["num_epochs"],
                                                                     cost_epochs=math.ceil(
                                                                         loss_epochs / self.options["num_epochs"])))

            self.trained = True


    def save(self, global_step):
        saver = tf.train.Saver()
        mdir = "%s/%s" % (MODELDIR, self.modelname)
        os.makedirs(mdir, exist_ok=True)
        saver.save(self.sess, os.path.join(mdir, self.modelname), global_step=global_step)
        json.dump(self.options, open(os.path.join(mdir, "attn_%s.params" % self.time), "w"))

    def transform(self, data, mask):
        # TODO
        encoder_inputs = "encoder_inputs:0"
        encoder_length = "encoder_length:0"

        if not self.trained:
            self.build_model()
            self.sess.run(tf.global_variables_initializer())

        feed = {
            encoder_inputs: data,
            encoder_length: [len(_) for _ in data]
        }

        return self.sess.run([self.inference_attention_matrices], feed_dict=feed)

    def variable_summaries(self, v):
        with tf.name_scope("summaries"):
            mean  = tf.reduce_mean(v)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(v - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(v))
            tf.summary.scalar('min', tf.reduce_min(v))
            tf.summary.histogram('histogram', v)


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("-n", dest="num_epochs", type=int, default=10)
    parse.add_argument("-b", dest="batch_size", type=int, default=100)
    parse.add_argument("-e", dest="embedding_dim", type=int, default=10)
    parse.add_argument("-t", dest="datetime", type=str, default=None)
    # parse.add_argument("-a", dest="attn_depth", type=int, default=100)
    parse.add_argument("-c", dest="cell_size", type=int, default=20)
    parse.add_argument("-V", dest="vocab_size", type=int, default=1)
    parse.add_argument("--lstm", dest="lstm", action="store_true", default=False)
    parse.add_argument("--gru", dest="gru", action="store_true", default=False)
    # parse.add_argument("-l", dest="max_seq_len", type=int)
    parse.add_argument("-L", dest="num_layers", type=int, default=2)
    parse.add_argument("--test", dest="test", action="store_true", default=False)
    parse.add_argument("--threshold", dest="threshold", help="rss memory threshold (G) [default: 100.0G]", type=float, default=100.0)
    parse.add_argument("--max-threshold", dest="max_threshold",help="max rss memory threshold updated to (G) [default: 200.0G]", type=float, default=200.0)
    parse.add_argument("--summaries", dest="is_summaries", help="if generate summaries", action="store_true", default=False)
    args = parse.parse_args()

    if args.test:
        data = np.tile(np.array([[1, 3, 2, 1], [1, 2, 3], [4, 2, 1], [2, 3, 4], [1, 2]]), [1])
        args.vocab_size = 4
    else:
        import pickle
        datafile = "../data/mseqs_code_by_patients.pkl"
        seqs = pickle.load(open(datafile, "rb"))
        data = []
        for a in seqs:
            for b in a:
                data.append(b)
        data = np.array(data)

        types = pickle.load(open("../data/mseqs_types_by_patients.pkl", "rb"))
        args.vocab_size = len(types)

    if args.datetime:
        options = json.load(
            open(os.path.join(MODELDIR, "attn_%s.model" % args.datetime, "attn_%s.params" % args.datetime), "r"))
        options["datetime"] = args.datetime
    else:
        _ = True if args.lstm or args.gru else False
        options = {
            "vocab_size": args.vocab_size + 1,
            "embedding_dim": args.embedding_dim,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "v": 1,
            "cell_size": args.cell_size,
            "LSTM": args.lstm if _ else True,
            "GRU": args.gru if _ else False,
            "num_layers": args.num_layers,
            "is_summaries": args.is_summaries
        }

    threshold = args.threshold
    max_threshold = args.max_threshold
    pid = os.getpid()

    model = AttentionModel(options)
    from mem_monitor import monitor
    monitor(model, "train", pid, threshold=threshold, max_threshold=max_threshold, params=(data,))