import tensorflow as tf
from data_processing import word_dict, process_seq, process_target, batch_iter
from Parameters import Parameters as pm
from tensorflow.python.util import nest

class Seq2Seq(object):

    def __init__(self):
        self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input')
        self.encoder_length = tf.placeholder(tf.int32, [None], name='encoder_length')
        self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decoder_input')
        self.decoder_length = tf.placeholder(tf.int32, [None], name='decoder_length')
        self.decoder_maxlength = tf.reduce_max(self.decoder_length, name='max_target_len')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.mask = tf.sequence_mask(self.decoder_length, self.decoder_maxlength, dtype=tf.float32, name='mask')
        self.keep_pro = tf.placeholder(tf.float32, name='keep_pro')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.chatbot_seq2seq()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，使用一个single_rnn_cell的函数，如果cells直接放入MultiRNNCell会报错。
            single_cell = tf.contrib.rnn.LSTMCell(pm.hidden_dim)
            #添加dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=pm.keep_pro)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(pm.num_layers)])
        return cell

    def chatbot_seq2seq(self):

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_size])
            self.embedding_input = tf.nn.embedding_lookup(self.embedding, self.encoder_input)

        with tf.variable_scope('encoder'):
            Cell = self._create_rnn_cell()
            self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(Cell, self.embedding_input, sequence_length=self.encoder_length,
                                                                        dtype=tf.float32)

        with tf.variable_scope('decoder'):
            self.decodercell = self._create_rnn_cell()
            #定义attention机制
            self.decodercellattention, self.decoder_initial_state = self.Attentioncell(False)
            self.output_layer = tf.layers.Dense(pm.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        with tf.variable_scope('train_decode'): #训练阶段
        # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>，并删除最后一位
            slice = tf.strided_slice(self.decoder_input, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_ = tf.concat([tf.fill([self.batch_size, 1], word_dict['<GO>']), slice], 1)
            decoder_input = tf.nn.embedding_lookup(self.embedding, decoder_)
            train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input, sequence_length=self.decoder_length,
                                                             time_major=False, name='train_helper')
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decodercellattention, helper=train_helper,
                                                            initial_state=self.decoder_initial_state, output_layer=self.output_layer)
            decoder_output, _,  _ = tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder, impute_finished=True,
                                                                      maximum_iterations=self.decoder_maxlength)
            self.decoder_train_logits = decoder_output.rnn_output
            #self.decoder_train_predict = tf.argmax(self.decoder_train_logits, axis=-1, name='train_predict')

        with tf.variable_scope('loss'):
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_train_logits, targets=self.decoder_input,
                                                         weights=self.mask)
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.variable_scope('predict_decode'): #预测阶段，共享训练阶段参数
            decodercell, decoder_initial_state = self.Attentioncell(True)
            start_token = tf.ones([self.batch_size, ], tf.int32) * word_dict['<GO>']
            end_token = word_dict['<EOS>']
            infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decodercell,
                                                                 embedding=self.embedding,
                                                                 start_tokens=start_token, end_token=end_token,
                                                                 initial_state=decoder_initial_state,
                                                                 beam_width=pm.beam_size,
                                                                 output_layer=self.output_layer)
            decoder_outputs, _,  _ = tf.contrib.seq2seq.dynamic_decode(decoder=infer_decoder, maximum_iterations=10)
            self.decoder_predict_decode = decoder_outputs.predicted_ids

    def Attentioncell(self, beam_search):
        if beam_search == True:
            encoder_output = tf.contrib.seq2seq.tile_batch(self.encoder_output, multiplier=pm.beam_size)
            encoder_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=pm.beam_size)
            encoder_length = tf.contrib.seq2seq.tile_batch(self.encoder_length, multiplier=pm.beam_size)
            batch_size = self.batch_size * pm.beam_size
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=pm.hidden_dim,
                                                                       memory=encoder_output,
                                                                       memory_sequence_length=encoder_length)
            decodercell = tf.contrib.seq2seq.AttentionWrapper(cell=self.decodercell,
                                                              attention_mechanism=attention_mechanism,
                                                              attention_layer_size=pm.hidden_dim)

            decoder_initial_state = decodercell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)

        elif beam_search == False:
            # 定义attention机制
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=pm.hidden_dim,
                                                                       memory=self.encoder_output,
                                                                       memory_sequence_length=self.encoder_length)
            # 定义使用LStmCell,封装attention。

            decodercell = tf.contrib.seq2seq.AttentionWrapper(cell=self.decodercell,
                                                              attention_mechanism=attention_mechanism,
                                                              attention_layer_size=pm.hidden_dim,
                                                              name='AttentionWrapper')

            decoder_initial_state = decodercell.zero_state(batch_size=self.batch_size,
                                                           dtype=tf.float32).clone(cell_state=self.encoder_state)

        return decodercell, decoder_initial_state


    def feed_data(self, x_batch, y_batch, keep_pro):
        x_pad, x_length = process_seq(x_batch)
        y_target, y_length = process_target(y_batch)
        feed_dict = {self.encoder_input: x_pad,
                     self.encoder_length: x_length,
                     self.decoder_input: y_target,
                     self.decoder_length: y_length,
                     self.batch_size: len(x_pad),
                     self.keep_pro: keep_pro}
        return feed_dict

    def test(self, sess, x, y):
        batch_test = batch_iter(x, y, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_test:
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            test_loss = sess.run(self.loss, feed_dict=feed_dict)
        return test_loss


    def predict(self, sess, x_batch):
        x_pad, x_length = process_seq(x_batch)
        predicted = sess.run(self.decoder_predict_decode, feed_dict={self.encoder_input: x_pad,
                                                                     self.encoder_length: x_length,
                                                                     self.batch_size: len(x_pad),
                                                                     self.keep_pro: 1.0})
        return predicted