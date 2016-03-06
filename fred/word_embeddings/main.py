import tensorflow as tf

EMBEDDING_SIZE = 8
BATCH_SIZE = 2

import input_ebooks
data = input_ebooks.read_data_sets()

batch_size = 2
embedding_size = 100
embedding_size_sqroot = 10
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
num_sampled = 64 # Number of negative examples to sample.
num_steps = 100001

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    x = tf.placeholder(tf.int32, shape=[batch_size], name="Input")
    y = tf.placeholder(tf.int32, shape=[batch_size, 1], name="Output")

    embeddings = tf.Variable(tf.random_uniform([data.dictionary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([data.dictionary_size, embedding_size], stddev=1.0 / embedding_size_sqroot))
    softmax_biases = tf.Variable(tf.zeros([data.dictionary_size]))

    embed = tf.nn.embedding_lookup(embeddings, x)
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed, y, num_sampled, data.dictionary_size))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print 'Initialized'
    average_loss = 0.
    for step in range(num_steps):
        batch_x, batch_y = data.get_batch(batch_size, num_skips, skip_window)
        _, loss_value = session.run([optimizer, loss], feed_dict = { x: batch_x, y: batch_y })

        average_loss += loss_value / 1000

        if step % 1000 == 0:
            print 'Loss at step %d (%d%%): %f' % (step, (step * 100 / num_steps), loss_value)
            average_loss = 0.

    #final_embeddings = normalized_embeddings.eval()

