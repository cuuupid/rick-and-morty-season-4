import numpy as np
import tensorflow as tf
import glob
from console_logging.console import Console
console = Console()

console.timeless()
console.monotone()

script_filenames = sorted(glob.glob("./data/*.txt"))

console.info("Found %d scripts." % len(script_filenames))

import codecs

corpus_raw = u""
for filename in script_filenames:
    with codecs.open(filename, 'r', 'utf-8') as script_file:
        # '''
        lines = script_file.readlines()
        cleaned = []
        remove_by = lines[0]
        for line in lines:
            if remove_by not in line:
                cleaned.append(line)
        cleaned = [remove_by] + cleaned
        corpus_raw += ''.join(cleaned)
        # '''
        # corpus_raw += script_file.read()

console.info("Corpus is %d characters long."%len(corpus_raw))

def create_lookup_tables(text):
    vocab = set(text)
    one_hot_decode = {key: word for key, word in enumerate(vocab)}
    one_hot_encode = {word: key for key, word in enumerate(vocab)}
    return one_hot_encode, one_hot_decode

def token_lookup():
    return {
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'   
    }

import pickle

token_dict = token_lookup()

for token, replacement in token_dict.items():
    corpus_raw = corpus_raw.replace(token, ' %s '%replacement)
corpus_raw = corpus_raw.lower()
corpus_raw = corpus_raw.split()

encoder, decoder = create_lookup_tables(corpus_raw)

corpus_vector = [encoder[word] for word in corpus_raw]
pickle.dump((corpus_vector, encoder, decoder, token_dict), open('preprocess.pkl','wb'))

def get_batches(int_text, batch_size, seq_length):
    words_per_batch = batch_size * seq_length
    num_batches = len(int_text)//words_per_batch
    int_text = int_text[:num_batches*words_per_batch]
    y = np.array(int_text[1:] + [int_text[0]])
    x = np.array(int_text)
    
    x_batches = np.split(x.reshape(batch_size, -1), num_batches, axis=1)
    y_batches = np.split(y.reshape(batch_size, -1), num_batches, axis=1)
    
    batch_data = list(zip(x_batches, y_batches))
    
    return np.array(batch_data)

num_epochs = 10000
batch_size = 512
rnn_size = 512
num_layers = 3
keep_prob = 0.7
embed_dim = 512
seq_length = 30
learning_rate = 0.001
save_dir = './save'

train_graph = tf.Graph()
with train_graph.as_default():    
    input_text = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    vocab_size = len(decoder)
    input_text_shape = tf.shape(input_text)
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * num_layers)
    initial_state = cell.zero_state(input_text_shape[0], tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    embed = tf.contrib.layers.embed_sequence(input_text, vocab_size, embed_dim)
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    probs = tf.nn.softmax(logits, name='probs')
    cost = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_text_shape[0], input_text_shape[1]])
    )
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

import time

pickle.dump((seq_length, save_dir), open('params.pkl', 'wb'))
batches = get_batches(corpus_vector, batch_size, seq_length)
num_batches = len(batches)
start_time = time.time()

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})
        for batch_index, (x, y) in enumerate(batches):
            feed_dict = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate
            }
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed_dict)
        time_elapsed = time.time() - start_time
        console.log('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}   time_elapsed = {:.3f}   time_remaining = {:.0f}'.format(
            epoch + 1,
            batch_index + 1,
            len(batches),
            train_loss,
            time_elapsed,
            ((num_batches * num_epochs)/((epoch + 1) * (batch_index + 1))) * time_elapsed - time_elapsed))
        if epoch % 10 == 0:
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            console.success('Model Trained and Saved')
