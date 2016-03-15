import tensorflow as tf
import math

EMBEDDING_SIZE = 100
BATCH_SIZE = 400

import input_ebooks
data = input_ebooks.read_data_sets()

batch_size = BATCH_SIZE
embedding_size = EMBEDDING_SIZE
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
num_sampled = 64 # Number of negative examples to sample.
num_steps = 100001

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    x = tf.placeholder(tf.int32, shape=[batch_size], name="Input")
    y = tf.placeholder(tf.int32, shape=[batch_size, 1], name="Output")

    embeddings = tf.Variable(tf.random_uniform([data.vocabulary_size, embedding_size], -1.0, 1.0), name="Embeddings")
    embed = tf.nn.embedding_lookup(embeddings, x)

    nce_weights = tf.Variable(tf.truncated_normal([data.vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)), name="NCE_Weights")
    nce_biases = tf.Variable(tf.zeros([data.vocabulary_size]), name="NCE_Biases")

    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, y, num_sampled, data.vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print 'Initialized'

    average_loss = 0.
    for step in xrange(num_steps):
        batch_x, batch_y = data.get_batch(batch_size, num_skips, skip_window)
        _, loss_value = session.run([optimizer, loss], feed_dict = { x: batch_x, y: batch_y })
        average_loss += loss_value / 1000

        if step % 1000 == 0:
            print 'Loss at step %d (%d%%): %f' % (step, (step * 100 / num_steps), loss_value)
            average_loss = 0.

    #final_embeddings = normalized_embeddings.eval()

