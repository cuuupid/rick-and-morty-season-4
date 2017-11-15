import tensorflow as tf
import numpy as np
import pickle

corpus_vector, encoder, decoder, token_dict = pickle.load(open('preprocess.pkl','rb'))
seq_length, save_dir = pickle.load(open('params.pkl','rb'))

def pick_word(probs, decoder):
    return np.random.choice(list(decoder.values()), 1, p=probs)[0]

gen_length = 1000
prime_words = 'rick'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)
    input_text = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    gen_sentences = prime_words.split()
    prev_state = sess.run(initial_state, {input_text: np.array([[1 for word in gen_sentences]])})
    
    for n in range(gen_length):
        dyn_input = [[encoder[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        pred_word = pick_word(probabilities[0,dyn_seq_length-1], decoder)
        gen_sentences.append(pred_word)
        
    episode_text = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        episode_text = episode_text.replace(' ' + token.lower(), key)
        
    print(episode_text)

episode_text = ' '.join(gen_sentences)
for key, token in token_dict.items():
    episode_text = episode_text.replace(' ' + token.lower(), key)
episode_text = episode_text.replace('\n ', '\n')
episode_text = episode_text.replace('( ', '(')
episode_text = episode_text.replace(' ”', '”')

import os
version_dir = './generated-episodes'
if not os.path.exists(version_dir): os.makedirs(version_dir)

num_episodes = len([name for name in os.listdir(version_dir) if os.path.isfile(os.path.join(version_dir, name))])
next_episode = version_dir + '/episode-' + str(num_episodes + 1) + '.md'
with open(next_episode, 'w') as text_file: text_file.write(episode_text)
