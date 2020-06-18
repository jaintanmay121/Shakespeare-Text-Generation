import numpy as np
import tensorflow as tf
import re

text = open('shakespeare.txt').read()


vocab = sorted(set(text))    
char2index = {char: index for index, char in enumerate(vocab)}  
index2char = np.array(vocab)       


def generate_text(model, start_string="Romeo:", num_generate = 1000, temperature=1.0):
    
    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    text_generated = []
    
    for char_index in range(num_generate):
        predictions = model(input_indices)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = str(tf.random.categorical(predictions,num_samples=1)[-1,0])
        s=re.search('\(', predicted_id).start()+1
        e=re.search(',', predicted_id).start()
        predicted_id=int(predicted_id[s:e])

        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    return (start_string + ''.join(text_generated))
