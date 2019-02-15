import os
import tensorflow as tf
from Parameters import Parameters as pm
from data_processing import label2id, batch_iter
from seq2seq import Seq2Seq

def train():

    tensorboard_dir = './tensorboard/Seq2Seq'
    save_dir = './checkpoints/Seq2Seq'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    tf.summary.scalar('loss', model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    x, y = label2id(pm.train_data)
    x_train, y_train = x[1:100001], y[1:100001]
    x_test, y_test = x[50001:80000], y[50001:80000]
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch+1)
        num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_train, y_train, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = model.feed_data(x_batch, y_batch, pm.keep_pro)
            _, global_step, _summary, train_loss = session.run([model.optimizer, model.global_step, merged_summary,
                                                                model.loss], feed_dict=feed_dict)
            if global_step % 100 == 0:
                test_loss = model.test(session, x_test, y_test)
                print('global_step:', global_step, 'train_loss:', train_loss, 'test_loss:', test_loss)

            if global_step % (3*num_batchs) == 0:
                print('Saving Model...')
                saver.save(session, save_path, global_step=global_step)
        pm.learning_rate *= pm.lr


if __name__ == '__main__':
    pm = pm
    model = Seq2Seq()
    train()
