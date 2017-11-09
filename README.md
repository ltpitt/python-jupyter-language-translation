
# Language Translation
This project is a peek into the realm of neural network machine translation.  We’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, we will be using just a small portion of the English corpus.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Let's play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028
    
    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .
    
    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .
    

## Implement Preprocessing Function
### Text to Word Ids
As common practice with other RNNs, we must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, we'll turn `source_text` and `target_text` from words to ids.  However, we also need to add the `<EOS>` word id at the end of `target_text`.  This will help the neural network predict when the sentence should end.

We can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
We can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    source_id_text = [[source_vocab_to_int[word] for word in n.split()] for n in source_text.split("\n")]
    target_id_text = [[target_vocab_to_int[word] for word in n.split()]+[target_vocab_to_int['<EOS>']] for n in target_text.split("\n")]
    return source_id_text, target_id_text

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    Tests Passed
    

### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# Check Point
This is our first checkpoint. If we ever decide to come back to this notebook or have to restart the notebook, we can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure we have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.1.0
    

    C:\Users\Pitto\Anaconda3\envs\py3\lib\site-packages\ipykernel_launcher.py:15: UserWarning: No GPU found. Please use a GPU to train your neural network.
      from ipykernel import kernelapp as app
    

## Build the Neural Network
We'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoder_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implemented the `model_inputs()` function to create TF Placeholders for the Neural Network. It creates the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
- Target sequence length placeholder named "target_sequence_length" with rank 1
- Max target sequence length tensor named "max_target_len" getting its value from applying tf.reduce_max on the target_sequence_length placeholder. Rank 0.
- Source sequence length placeholder named "source_sequence_length" with rank 1

Returns the placeholders in the following the tuple (input, targets, learning rate, keep probability, target sequence length, max target sequence length, source sequence length)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    input = tf.placeholder(tf.int32,[None,None],name='input')
    target = tf.placeholder(tf.int32,[None,None])
    target_sequence_len = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_len,name='max_target_length' )
    source_sequence_len = tf.placeholder(tf.int32, [None], name='source_sequence_length')
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return input, target, learning_rate, keep_prob, target_sequence_len, max_target_len, source_sequence_len

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed
    

### Process Decoder Input
Implemented `process_decoder_input` by removing the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


```python
def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
    return decoder_input

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_encoding_input(process_decoder_input)
```

    Tests Passed
    

### Encoding
Implemented `encoding_layer()` to create a Encoder RNN layer:
 * Embed the encoder input using [`tf.contrib.layers.embed_sequence`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence)
 * Construct a [stacked](https://github.com/tensorflow/tensorflow/blob/6947f65a374ebf29e74bb71e36fd82760056d82c/tensorflow/docs_src/tutorials/recurrent.md#stacking-multiple-lstms) [`tf.contrib.rnn.LSTMCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell) wrapped in a [`tf.contrib.rnn.DropoutWrapper`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper)
 * Pass cell and embedded input to [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)


```python
from imp import reload
reload(tests)

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    encoder_inputs = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(rnn_size), output_keep_prob=keep_prob) for _ in range(num_layers)])
    RNN_output, RNN_state = tf.nn.dynamic_rnn(cell, encoder_inputs, sequence_length=source_sequence_length, dtype=tf.float32)
    return RNN_output, RNN_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed
    

### Decoding - Training
Creates a training decoding layer:
* Creates a [`tf.contrib.seq2seq.TrainingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper) 
* Creates a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtains the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


```python

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=target_sequence_length, time_major=False)
    basic_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,training_helper,encoder_state,output_layer)
    training_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(basic_decoder,impute_finished=True,maximum_iterations= max_summary_length)
    return training_decoder_output 



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed
    

### Decoding - Inference
Creates inference decoder:
* Creates a [`tf.contrib.seq2seq.GreedyEmbeddingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper)
* Creates a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtains the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    start_of_sequence_id = target_vocab_to_int['<GO>']
    end_of_sequence_id = target_vocab_to_int['<EOS>']
   
       
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')

       
       
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens,
                                                                    end_of_sequence_id)
       
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           inference_helper,
                                                           encoder_state,
                                                           output_layer)
       
    inference_decoder_output, _= tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations= max_target_sequence_length)

    return inference_decoder_output



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed
    

### Build the Decoding Layer
Implemented `decoding_layer()` to create a Decoder RNN layer.

* Embedded the target sequences
* Constructed the decoder LSTM cell (just like you constructed the encoder cell above)
* Created an output layer to map the outputs of the decoder to the elements of our vocabulary
* Useed the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)` function to get the training logits.
* Useed your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob)` function to get the inference logits.

Note: We need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
from tensorflow.python.layers import core as layers_core

def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # embedding target sequence
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    # construct decoder lstm cell
    dec_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size) for _ in range(num_layers)]),output_keep_prob=keep_prob)
    # create output layer to map the outputs of the decoder to the elements of our vocabulary
    output_layer = layers_core.Dense(target_vocab_size,
                                    kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    # decoder train
    with tf.variable_scope("decoding") as decoding_scope:
        dec_outputs_train = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                             target_sequence_length, max_target_sequence_length, 
                             output_layer, keep_prob)
    # decoder inference
    start_of_sequence_id = target_vocab_to_int["<GO>"]
    end_of_sequence_id = target_vocab_to_int["<EOS>"]
    with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        dec_outputs_infer = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                             end_of_sequence_id, max_target_sequence_length,
                             target_vocab_size, output_layer, batch_size, keep_prob)
    # rerturn
    return dec_outputs_train, dec_outputs_infer



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed
    

### Build the Neural Network
Applying the functions implemented above to:

- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,  source_sequence_length, source_vocab_size, encoding_embedding_size)`.
- Process target data using your `process_decoder_input(target_data, target_vocab_to_int, batch_size)` function.
- Decode the encoded input using your `decoding_layer(dec_input, enc_state, target_sequence_length, max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size)` function.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_output, enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   enc_embedding_size)
    # process target data
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size) 
    # embedding and decoding
    dec_outputs_train, dec_outputs_infer = decoding_layer(dec_input, enc_state,
                   target_sequence_length, tf.reduce_max(max_target_sentence_length),
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, dec_embedding_size)
    return dec_outputs_train, dec_outputs_infer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed
    

## Neural Network Training
### Hyperparameters
Tune the following parameters to change NN behaviour:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability
- Set `display_step` to state how many steps between each debug output statement


```python
# Number of Epochs
epochs = 5
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = 0.01
# Dropout Keep Probability
keep_probability = 0.8
# Display Step
display_step = 10

```

### Building the Graph
Building the graph using the neural network we implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

```

Batch and pad the source and target sequences


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

```

### Train
Training the neural network on the preprocessed data.  


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))                                                                                                  
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch   10/538 - Train Accuracy: 0.2686, Validation Accuracy: 0.3530, Loss: 3.5477
    Epoch   0 Batch   20/538 - Train Accuracy: 0.3469, Validation Accuracy: 0.3858, Loss: 2.8958
    Epoch   0 Batch   30/538 - Train Accuracy: 0.3609, Validation Accuracy: 0.4206, Loss: 2.7392
    Epoch   0 Batch   40/538 - Train Accuracy: 0.4526, Validation Accuracy: 0.4474, Loss: 2.2537
    Epoch   0 Batch   50/538 - Train Accuracy: 0.4592, Validation Accuracy: 0.5014, Loss: 2.1874
    Epoch   0 Batch   60/538 - Train Accuracy: 0.4574, Validation Accuracy: 0.5158, Loss: 2.0013
    Epoch   0 Batch   70/538 - Train Accuracy: 0.4933, Validation Accuracy: 0.5217, Loss: 1.7485
    Epoch   0 Batch   80/538 - Train Accuracy: 0.4713, Validation Accuracy: 0.5220, Loss: 1.6875
    Epoch   0 Batch   90/538 - Train Accuracy: 0.5179, Validation Accuracy: 0.5384, Loss: 1.3981
    Epoch   0 Batch  100/538 - Train Accuracy: 0.4977, Validation Accuracy: 0.5236, Loss: 1.2250
    Epoch   0 Batch  110/538 - Train Accuracy: 0.5145, Validation Accuracy: 0.5447, Loss: 1.1504
    Epoch   0 Batch  120/538 - Train Accuracy: 0.5012, Validation Accuracy: 0.5444, Loss: 0.9934
    Epoch   0 Batch  130/538 - Train Accuracy: 0.5147, Validation Accuracy: 0.5323, Loss: 0.8930
    Epoch   0 Batch  140/538 - Train Accuracy: 0.5141, Validation Accuracy: 0.5558, Loss: 0.9148
    Epoch   0 Batch  150/538 - Train Accuracy: 0.5449, Validation Accuracy: 0.5501, Loss: 0.8247
    Epoch   0 Batch  160/538 - Train Accuracy: 0.5552, Validation Accuracy: 0.5506, Loss: 0.7543
    Epoch   0 Batch  170/538 - Train Accuracy: 0.5863, Validation Accuracy: 0.5636, Loss: 0.7417
    Epoch   0 Batch  180/538 - Train Accuracy: 0.6047, Validation Accuracy: 0.5815, Loss: 0.6984
    Epoch   0 Batch  190/538 - Train Accuracy: 0.6153, Validation Accuracy: 0.6003, Loss: 0.6839
    Epoch   0 Batch  200/538 - Train Accuracy: 0.6086, Validation Accuracy: 0.6060, Loss: 0.6629
    Epoch   0 Batch  210/538 - Train Accuracy: 0.5869, Validation Accuracy: 0.5875, Loss: 0.6389
    Epoch   0 Batch  220/538 - Train Accuracy: 0.5932, Validation Accuracy: 0.5895, Loss: 0.6169
    Epoch   0 Batch  230/538 - Train Accuracy: 0.6188, Validation Accuracy: 0.6314, Loss: 0.6339
    Epoch   0 Batch  240/538 - Train Accuracy: 0.6227, Validation Accuracy: 0.6108, Loss: 0.6212
    Epoch   0 Batch  250/538 - Train Accuracy: 0.6289, Validation Accuracy: 0.6223, Loss: 0.5924
    Epoch   0 Batch  260/538 - Train Accuracy: 0.6291, Validation Accuracy: 0.6557, Loss: 0.5761
    Epoch   0 Batch  270/538 - Train Accuracy: 0.6504, Validation Accuracy: 0.6614, Loss: 0.5793
    Epoch   0 Batch  280/538 - Train Accuracy: 0.6734, Validation Accuracy: 0.6491, Loss: 0.5349
    Epoch   0 Batch  290/538 - Train Accuracy: 0.6643, Validation Accuracy: 0.6628, Loss: 0.5370
    Epoch   0 Batch  300/538 - Train Accuracy: 0.6750, Validation Accuracy: 0.6781, Loss: 0.5152
    Epoch   0 Batch  310/538 - Train Accuracy: 0.6947, Validation Accuracy: 0.6921, Loss: 0.5109
    Epoch   0 Batch  320/538 - Train Accuracy: 0.6888, Validation Accuracy: 0.6926, Loss: 0.4903
    Epoch   0 Batch  330/538 - Train Accuracy: 0.6778, Validation Accuracy: 0.6614, Loss: 0.4667
    Epoch   0 Batch  340/538 - Train Accuracy: 0.6771, Validation Accuracy: 0.6836, Loss: 0.4912
    Epoch   0 Batch  350/538 - Train Accuracy: 0.7065, Validation Accuracy: 0.6939, Loss: 0.4590
    Epoch   0 Batch  360/538 - Train Accuracy: 0.6912, Validation Accuracy: 0.7108, Loss: 0.4437
    Epoch   0 Batch  370/538 - Train Accuracy: 0.7111, Validation Accuracy: 0.7131, Loss: 0.4454
    Epoch   0 Batch  380/538 - Train Accuracy: 0.7559, Validation Accuracy: 0.7314, Loss: 0.4108
    Epoch   0 Batch  390/538 - Train Accuracy: 0.7338, Validation Accuracy: 0.7085, Loss: 0.3934
    Epoch   0 Batch  400/538 - Train Accuracy: 0.7180, Validation Accuracy: 0.7095, Loss: 0.3924
    Epoch   0 Batch  410/538 - Train Accuracy: 0.7432, Validation Accuracy: 0.7280, Loss: 0.3879
    Epoch   0 Batch  420/538 - Train Accuracy: 0.7344, Validation Accuracy: 0.7049, Loss: 0.3823
    Epoch   0 Batch  430/538 - Train Accuracy: 0.7375, Validation Accuracy: 0.7156, Loss: 0.3530
    Epoch   0 Batch  440/538 - Train Accuracy: 0.7438, Validation Accuracy: 0.7360, Loss: 0.3512
    Epoch   0 Batch  450/538 - Train Accuracy: 0.7359, Validation Accuracy: 0.7244, Loss: 0.3419
    Epoch   0 Batch  460/538 - Train Accuracy: 0.7480, Validation Accuracy: 0.7573, Loss: 0.3144
    Epoch   0 Batch  470/538 - Train Accuracy: 0.7660, Validation Accuracy: 0.7495, Loss: 0.3089
    Epoch   0 Batch  480/538 - Train Accuracy: 0.7826, Validation Accuracy: 0.7775, Loss: 0.2863
    Epoch   0 Batch  490/538 - Train Accuracy: 0.7984, Validation Accuracy: 0.7995, Loss: 0.2710
    Epoch   0 Batch  500/538 - Train Accuracy: 0.8349, Validation Accuracy: 0.7905, Loss: 0.2391
    Epoch   0 Batch  510/538 - Train Accuracy: 0.8263, Validation Accuracy: 0.8042, Loss: 0.2474
    Epoch   0 Batch  520/538 - Train Accuracy: 0.7877, Validation Accuracy: 0.8086, Loss: 0.2489
    Epoch   0 Batch  530/538 - Train Accuracy: 0.7967, Validation Accuracy: 0.8184, Loss: 0.2509
    Epoch   1 Batch   10/538 - Train Accuracy: 0.8295, Validation Accuracy: 0.8265, Loss: 0.2437
    Epoch   1 Batch   20/538 - Train Accuracy: 0.8354, Validation Accuracy: 0.8345, Loss: 0.2117
    Epoch   1 Batch   30/538 - Train Accuracy: 0.8449, Validation Accuracy: 0.8530, Loss: 0.2079
    Epoch   1 Batch   40/538 - Train Accuracy: 0.8484, Validation Accuracy: 0.8638, Loss: 0.1758
    Epoch   1 Batch   50/538 - Train Accuracy: 0.8684, Validation Accuracy: 0.8697, Loss: 0.1792
    Epoch   1 Batch   60/538 - Train Accuracy: 0.8436, Validation Accuracy: 0.8485, Loss: 0.1796
    Epoch   1 Batch   70/538 - Train Accuracy: 0.8785, Validation Accuracy: 0.8537, Loss: 0.1685
    Epoch   1 Batch   80/538 - Train Accuracy: 0.8742, Validation Accuracy: 0.8810, Loss: 0.1764
    Epoch   1 Batch   90/538 - Train Accuracy: 0.8728, Validation Accuracy: 0.8665, Loss: 0.1729
    Epoch   1 Batch  100/538 - Train Accuracy: 0.8797, Validation Accuracy: 0.8700, Loss: 0.1473
    Epoch   1 Batch  110/538 - Train Accuracy: 0.8697, Validation Accuracy: 0.8707, Loss: 0.1598
    Epoch   1 Batch  120/538 - Train Accuracy: 0.8965, Validation Accuracy: 0.8983, Loss: 0.1258
    Epoch   1 Batch  130/538 - Train Accuracy: 0.8687, Validation Accuracy: 0.8869, Loss: 0.1378
    Epoch   1 Batch  140/538 - Train Accuracy: 0.8727, Validation Accuracy: 0.8945, Loss: 0.1474
    Epoch   1 Batch  150/538 - Train Accuracy: 0.8996, Validation Accuracy: 0.8952, Loss: 0.1254
    Epoch   1 Batch  160/538 - Train Accuracy: 0.8824, Validation Accuracy: 0.8915, Loss: 0.1177
    Epoch   1 Batch  170/538 - Train Accuracy: 0.8945, Validation Accuracy: 0.8883, Loss: 0.1253
    Epoch   1 Batch  180/538 - Train Accuracy: 0.9118, Validation Accuracy: 0.8833, Loss: 0.1180
    Epoch   1 Batch  190/538 - Train Accuracy: 0.9022, Validation Accuracy: 0.9228, Loss: 0.1296
    Epoch   1 Batch  200/538 - Train Accuracy: 0.9053, Validation Accuracy: 0.9015, Loss: 0.0992
    Epoch   1 Batch  210/538 - Train Accuracy: 0.9023, Validation Accuracy: 0.9102, Loss: 0.1065
    Epoch   1 Batch  220/538 - Train Accuracy: 0.8977, Validation Accuracy: 0.8960, Loss: 0.1068
    Epoch   1 Batch  230/538 - Train Accuracy: 0.8939, Validation Accuracy: 0.8991, Loss: 0.1004
    Epoch   1 Batch  240/538 - Train Accuracy: 0.9062, Validation Accuracy: 0.9283, Loss: 0.0942
    Epoch   1 Batch  250/538 - Train Accuracy: 0.9158, Validation Accuracy: 0.9025, Loss: 0.0928
    Epoch   1 Batch  260/538 - Train Accuracy: 0.8897, Validation Accuracy: 0.8922, Loss: 0.0985
    Epoch   1 Batch  270/538 - Train Accuracy: 0.9146, Validation Accuracy: 0.8961, Loss: 0.0865
    Epoch   1 Batch  280/538 - Train Accuracy: 0.9224, Validation Accuracy: 0.9185, Loss: 0.0831
    Epoch   1 Batch  290/538 - Train Accuracy: 0.9295, Validation Accuracy: 0.9066, Loss: 0.0773
    Epoch   1 Batch  300/538 - Train Accuracy: 0.9139, Validation Accuracy: 0.9089, Loss: 0.0850
    Epoch   1 Batch  310/538 - Train Accuracy: 0.9293, Validation Accuracy: 0.9116, Loss: 0.0869
    Epoch   1 Batch  320/538 - Train Accuracy: 0.9087, Validation Accuracy: 0.9105, Loss: 0.0792
    Epoch   1 Batch  330/538 - Train Accuracy: 0.9276, Validation Accuracy: 0.9146, Loss: 0.0716
    Epoch   1 Batch  340/538 - Train Accuracy: 0.9238, Validation Accuracy: 0.9199, Loss: 0.0729
    Epoch   1 Batch  350/538 - Train Accuracy: 0.9262, Validation Accuracy: 0.9277, Loss: 0.0827
    Epoch   1 Batch  360/538 - Train Accuracy: 0.9064, Validation Accuracy: 0.9308, Loss: 0.0755
    Epoch   1 Batch  370/538 - Train Accuracy: 0.9111, Validation Accuracy: 0.9228, Loss: 0.0723
    Epoch   1 Batch  380/538 - Train Accuracy: 0.9221, Validation Accuracy: 0.9364, Loss: 0.0674
    Epoch   1 Batch  390/538 - Train Accuracy: 0.9364, Validation Accuracy: 0.9048, Loss: 0.0620
    Epoch   1 Batch  400/538 - Train Accuracy: 0.9234, Validation Accuracy: 0.9002, Loss: 0.0749
    Epoch   1 Batch  410/538 - Train Accuracy: 0.9344, Validation Accuracy: 0.9343, Loss: 0.0689
    Epoch   1 Batch  420/538 - Train Accuracy: 0.9449, Validation Accuracy: 0.9364, Loss: 0.0644
    Epoch   1 Batch  430/538 - Train Accuracy: 0.9156, Validation Accuracy: 0.9162, Loss: 0.0662
    Epoch   1 Batch  440/538 - Train Accuracy: 0.9227, Validation Accuracy: 0.9304, Loss: 0.0769
    Epoch   1 Batch  450/538 - Train Accuracy: 0.9111, Validation Accuracy: 0.9327, Loss: 0.0756
    Epoch   1 Batch  460/538 - Train Accuracy: 0.8966, Validation Accuracy: 0.9183, Loss: 0.0720
    Epoch   1 Batch  470/538 - Train Accuracy: 0.9420, Validation Accuracy: 0.9208, Loss: 0.0602
    Epoch   1 Batch  480/538 - Train Accuracy: 0.9349, Validation Accuracy: 0.9180, Loss: 0.0636
    Epoch   1 Batch  490/538 - Train Accuracy: 0.9446, Validation Accuracy: 0.9334, Loss: 0.0537
    Epoch   1 Batch  500/538 - Train Accuracy: 0.9595, Validation Accuracy: 0.9357, Loss: 0.0487
    Epoch   1 Batch  510/538 - Train Accuracy: 0.9394, Validation Accuracy: 0.9293, Loss: 0.0538
    Epoch   1 Batch  520/538 - Train Accuracy: 0.9270, Validation Accuracy: 0.9205, Loss: 0.0609
    Epoch   1 Batch  530/538 - Train Accuracy: 0.9090, Validation Accuracy: 0.9311, Loss: 0.0641
    Epoch   2 Batch   10/538 - Train Accuracy: 0.9271, Validation Accuracy: 0.9062, Loss: 0.0622
    Epoch   2 Batch   20/538 - Train Accuracy: 0.9364, Validation Accuracy: 0.9316, Loss: 0.0569
    Epoch   2 Batch   30/538 - Train Accuracy: 0.9326, Validation Accuracy: 0.9258, Loss: 0.0591
    Epoch   2 Batch   40/538 - Train Accuracy: 0.9361, Validation Accuracy: 0.9368, Loss: 0.0468
    Epoch   2 Batch   50/538 - Train Accuracy: 0.9346, Validation Accuracy: 0.9332, Loss: 0.0578
    Epoch   2 Batch   60/538 - Train Accuracy: 0.9271, Validation Accuracy: 0.9268, Loss: 0.0542
    Epoch   2 Batch   70/538 - Train Accuracy: 0.9299, Validation Accuracy: 0.9368, Loss: 0.0512
    Epoch   2 Batch   80/538 - Train Accuracy: 0.9529, Validation Accuracy: 0.9434, Loss: 0.0532
    Epoch   2 Batch   90/538 - Train Accuracy: 0.9371, Validation Accuracy: 0.9453, Loss: 0.0570
    Epoch   2 Batch  100/538 - Train Accuracy: 0.9453, Validation Accuracy: 0.9387, Loss: 0.0467
    Epoch   2 Batch  110/538 - Train Accuracy: 0.9383, Validation Accuracy: 0.9416, Loss: 0.0568
    Epoch   2 Batch  120/538 - Train Accuracy: 0.9521, Validation Accuracy: 0.9240, Loss: 0.0404
    Epoch   2 Batch  130/538 - Train Accuracy: 0.9544, Validation Accuracy: 0.9240, Loss: 0.0516
    Epoch   2 Batch  140/538 - Train Accuracy: 0.9199, Validation Accuracy: 0.9311, Loss: 0.0645
    Epoch   2 Batch  150/538 - Train Accuracy: 0.9520, Validation Accuracy: 0.9419, Loss: 0.0411
    Epoch   2 Batch  160/538 - Train Accuracy: 0.9371, Validation Accuracy: 0.9444, Loss: 0.0436
    Epoch   2 Batch  170/538 - Train Accuracy: 0.9353, Validation Accuracy: 0.9444, Loss: 0.0548
    Epoch   2 Batch  180/538 - Train Accuracy: 0.9462, Validation Accuracy: 0.9460, Loss: 0.0535
    Epoch   2 Batch  190/538 - Train Accuracy: 0.9314, Validation Accuracy: 0.9366, Loss: 0.0602
    Epoch   2 Batch  200/538 - Train Accuracy: 0.9561, Validation Accuracy: 0.9329, Loss: 0.0375
    Epoch   2 Batch  210/538 - Train Accuracy: 0.9442, Validation Accuracy: 0.9359, Loss: 0.0497
    Epoch   2 Batch  220/538 - Train Accuracy: 0.9394, Validation Accuracy: 0.9379, Loss: 0.0459
    Epoch   2 Batch  230/538 - Train Accuracy: 0.9348, Validation Accuracy: 0.9435, Loss: 0.0465
    Epoch   2 Batch  240/538 - Train Accuracy: 0.9605, Validation Accuracy: 0.9469, Loss: 0.0409
    Epoch   2 Batch  250/538 - Train Accuracy: 0.9441, Validation Accuracy: 0.9418, Loss: 0.0469
    Epoch   2 Batch  260/538 - Train Accuracy: 0.9252, Validation Accuracy: 0.9448, Loss: 0.0483
    Epoch   2 Batch  270/538 - Train Accuracy: 0.9531, Validation Accuracy: 0.9311, Loss: 0.0423
    Epoch   2 Batch  280/538 - Train Accuracy: 0.9501, Validation Accuracy: 0.9334, Loss: 0.0362
    Epoch   2 Batch  290/538 - Train Accuracy: 0.9557, Validation Accuracy: 0.9457, Loss: 0.0418
    Epoch   2 Batch  300/538 - Train Accuracy: 0.9598, Validation Accuracy: 0.9380, Loss: 0.0428
    Epoch   2 Batch  310/538 - Train Accuracy: 0.9547, Validation Accuracy: 0.9533, Loss: 0.0472
    Epoch   2 Batch  320/538 - Train Accuracy: 0.9410, Validation Accuracy: 0.9425, Loss: 0.0475
    Epoch   2 Batch  330/538 - Train Accuracy: 0.9652, Validation Accuracy: 0.9570, Loss: 0.0359
    Epoch   2 Batch  340/538 - Train Accuracy: 0.9449, Validation Accuracy: 0.9462, Loss: 0.0465
    Epoch   2 Batch  350/538 - Train Accuracy: 0.9472, Validation Accuracy: 0.9409, Loss: 0.0512
    Epoch   2 Batch  360/538 - Train Accuracy: 0.9406, Validation Accuracy: 0.9517, Loss: 0.0377
    Epoch   2 Batch  370/538 - Train Accuracy: 0.9506, Validation Accuracy: 0.9355, Loss: 0.0383
    Epoch   2 Batch  380/538 - Train Accuracy: 0.9520, Validation Accuracy: 0.9400, Loss: 0.0381
    Epoch   2 Batch  390/538 - Train Accuracy: 0.9516, Validation Accuracy: 0.9466, Loss: 0.0341
    Epoch   2 Batch  400/538 - Train Accuracy: 0.9691, Validation Accuracy: 0.9615, Loss: 0.0443
    Epoch   2 Batch  410/538 - Train Accuracy: 0.9625, Validation Accuracy: 0.9446, Loss: 0.0410
    Epoch   2 Batch  420/538 - Train Accuracy: 0.9701, Validation Accuracy: 0.9442, Loss: 0.0488
    Epoch   2 Batch  430/538 - Train Accuracy: 0.9414, Validation Accuracy: 0.9585, Loss: 0.0451
    Epoch   2 Batch  440/538 - Train Accuracy: 0.9488, Validation Accuracy: 0.9512, Loss: 0.0494
    Epoch   2 Batch  450/538 - Train Accuracy: 0.9390, Validation Accuracy: 0.9496, Loss: 0.0545
    Epoch   2 Batch  460/538 - Train Accuracy: 0.9379, Validation Accuracy: 0.9435, Loss: 0.0508
    Epoch   2 Batch  470/538 - Train Accuracy: 0.9648, Validation Accuracy: 0.9448, Loss: 0.0465
    Epoch   2 Batch  480/538 - Train Accuracy: 0.9591, Validation Accuracy: 0.9396, Loss: 0.0415
    Epoch   2 Batch  490/538 - Train Accuracy: 0.9420, Validation Accuracy: 0.9554, Loss: 0.0397
    Epoch   2 Batch  500/538 - Train Accuracy: 0.9684, Validation Accuracy: 0.9533, Loss: 0.0297
    Epoch   2 Batch  510/538 - Train Accuracy: 0.9682, Validation Accuracy: 0.9453, Loss: 0.0378
    Epoch   2 Batch  520/538 - Train Accuracy: 0.9473, Validation Accuracy: 0.9332, Loss: 0.0427
    Epoch   2 Batch  530/538 - Train Accuracy: 0.9273, Validation Accuracy: 0.9485, Loss: 0.0489
    Epoch   3 Batch   10/538 - Train Accuracy: 0.9520, Validation Accuracy: 0.9308, Loss: 0.0491
    Epoch   3 Batch   20/538 - Train Accuracy: 0.9494, Validation Accuracy: 0.9490, Loss: 0.0457
    Epoch   3 Batch   30/538 - Train Accuracy: 0.9561, Validation Accuracy: 0.9407, Loss: 0.0433
    Epoch   3 Batch   40/538 - Train Accuracy: 0.9498, Validation Accuracy: 0.9377, Loss: 0.0302
    Epoch   3 Batch   50/538 - Train Accuracy: 0.9637, Validation Accuracy: 0.9402, Loss: 0.0389
    Epoch   3 Batch   60/538 - Train Accuracy: 0.9479, Validation Accuracy: 0.9446, Loss: 0.0449
    Epoch   3 Batch   70/538 - Train Accuracy: 0.9494, Validation Accuracy: 0.9496, Loss: 0.0441
    Epoch   3 Batch   80/538 - Train Accuracy: 0.9520, Validation Accuracy: 0.9492, Loss: 0.0382
    Epoch   3 Batch   90/538 - Train Accuracy: 0.9613, Validation Accuracy: 0.9487, Loss: 0.0445
    Epoch   3 Batch  100/538 - Train Accuracy: 0.9613, Validation Accuracy: 0.9512, Loss: 0.0345
    Epoch   3 Batch  110/538 - Train Accuracy: 0.9592, Validation Accuracy: 0.9371, Loss: 0.0417
    Epoch   3 Batch  120/538 - Train Accuracy: 0.9742, Validation Accuracy: 0.9453, Loss: 0.0284
    Epoch   3 Batch  130/538 - Train Accuracy: 0.9516, Validation Accuracy: 0.9551, Loss: 0.0349
    Epoch   3 Batch  140/538 - Train Accuracy: 0.9424, Validation Accuracy: 0.9366, Loss: 0.0479
    Epoch   3 Batch  150/538 - Train Accuracy: 0.9557, Validation Accuracy: 0.9416, Loss: 0.0333
    Epoch   3 Batch  160/538 - Train Accuracy: 0.9481, Validation Accuracy: 0.9482, Loss: 0.0391
    Epoch   3 Batch  170/538 - Train Accuracy: 0.9464, Validation Accuracy: 0.9492, Loss: 0.0379
    Epoch   3 Batch  180/538 - Train Accuracy: 0.9585, Validation Accuracy: 0.9533, Loss: 0.0381
    Epoch   3 Batch  190/538 - Train Accuracy: 0.9416, Validation Accuracy: 0.9577, Loss: 0.0444
    Epoch   3 Batch  200/538 - Train Accuracy: 0.9729, Validation Accuracy: 0.9458, Loss: 0.0310
    Epoch   3 Batch  210/538 - Train Accuracy: 0.9401, Validation Accuracy: 0.9368, Loss: 0.0381
    Epoch   3 Batch  220/538 - Train Accuracy: 0.9418, Validation Accuracy: 0.9492, Loss: 0.0414
    Epoch   3 Batch  230/538 - Train Accuracy: 0.9402, Validation Accuracy: 0.9506, Loss: 0.0318
    Epoch   3 Batch  240/538 - Train Accuracy: 0.9494, Validation Accuracy: 0.9528, Loss: 0.0317
    Epoch   3 Batch  250/538 - Train Accuracy: 0.9637, Validation Accuracy: 0.9426, Loss: 0.0382
    Epoch   3 Batch  260/538 - Train Accuracy: 0.9405, Validation Accuracy: 0.9529, Loss: 0.0360
    Epoch   3 Batch  270/538 - Train Accuracy: 0.9527, Validation Accuracy: 0.9560, Loss: 0.0301
    Epoch   3 Batch  280/538 - Train Accuracy: 0.9663, Validation Accuracy: 0.9435, Loss: 0.0271
    Epoch   3 Batch  290/538 - Train Accuracy: 0.9684, Validation Accuracy: 0.9498, Loss: 0.0291
    Epoch   3 Batch  300/538 - Train Accuracy: 0.9533, Validation Accuracy: 0.9521, Loss: 0.0335
    Epoch   3 Batch  310/538 - Train Accuracy: 0.9678, Validation Accuracy: 0.9624, Loss: 0.0428
    Epoch   3 Batch  320/538 - Train Accuracy: 0.9531, Validation Accuracy: 0.9565, Loss: 0.0323
    Epoch   3 Batch  330/538 - Train Accuracy: 0.9712, Validation Accuracy: 0.9657, Loss: 0.0334
    Epoch   3 Batch  340/538 - Train Accuracy: 0.9555, Validation Accuracy: 0.9576, Loss: 0.0407
    Epoch   3 Batch  350/538 - Train Accuracy: 0.9647, Validation Accuracy: 0.9460, Loss: 0.0389
    Epoch   3 Batch  360/538 - Train Accuracy: 0.9387, Validation Accuracy: 0.9597, Loss: 0.0327
    Epoch   3 Batch  370/538 - Train Accuracy: 0.9545, Validation Accuracy: 0.9426, Loss: 0.0299
    Epoch   3 Batch  380/538 - Train Accuracy: 0.9543, Validation Accuracy: 0.9503, Loss: 0.0302
    Epoch   3 Batch  390/538 - Train Accuracy: 0.9340, Validation Accuracy: 0.9409, Loss: 0.0328
    Epoch   3 Batch  400/538 - Train Accuracy: 0.9585, Validation Accuracy: 0.9528, Loss: 0.0398
    Epoch   3 Batch  410/538 - Train Accuracy: 0.9736, Validation Accuracy: 0.9499, Loss: 0.0298
    Epoch   3 Batch  420/538 - Train Accuracy: 0.9605, Validation Accuracy: 0.9455, Loss: 0.0343
    Epoch   3 Batch  430/538 - Train Accuracy: 0.9506, Validation Accuracy: 0.9524, Loss: 0.0328
    Epoch   3 Batch  440/538 - Train Accuracy: 0.9559, Validation Accuracy: 0.9567, Loss: 0.0381
    Epoch   3 Batch  450/538 - Train Accuracy: 0.9425, Validation Accuracy: 0.9553, Loss: 0.0408
    Epoch   3 Batch  460/538 - Train Accuracy: 0.9436, Validation Accuracy: 0.9426, Loss: 0.0375
    Epoch   3 Batch  470/538 - Train Accuracy: 0.9697, Validation Accuracy: 0.9558, Loss: 0.0274
    Epoch   3 Batch  480/538 - Train Accuracy: 0.9652, Validation Accuracy: 0.9444, Loss: 0.0353
    Epoch   3 Batch  490/538 - Train Accuracy: 0.9671, Validation Accuracy: 0.9549, Loss: 0.0291
    Epoch   3 Batch  500/538 - Train Accuracy: 0.9766, Validation Accuracy: 0.9499, Loss: 0.0211
    Epoch   3 Batch  510/538 - Train Accuracy: 0.9637, Validation Accuracy: 0.9455, Loss: 0.0291
    Epoch   3 Batch  520/538 - Train Accuracy: 0.9574, Validation Accuracy: 0.9595, Loss: 0.0372
    Epoch   3 Batch  530/538 - Train Accuracy: 0.9436, Validation Accuracy: 0.9636, Loss: 0.0357
    Epoch   4 Batch   10/538 - Train Accuracy: 0.9441, Validation Accuracy: 0.9524, Loss: 0.0377
    Epoch   4 Batch   20/538 - Train Accuracy: 0.9580, Validation Accuracy: 0.9414, Loss: 0.0358
    Epoch   4 Batch   30/538 - Train Accuracy: 0.9471, Validation Accuracy: 0.9542, Loss: 0.0339
    Epoch   4 Batch   40/538 - Train Accuracy: 0.9647, Validation Accuracy: 0.9576, Loss: 0.0255
    Epoch   4 Batch   50/538 - Train Accuracy: 0.9666, Validation Accuracy: 0.9490, Loss: 0.0309
    Epoch   4 Batch   60/538 - Train Accuracy: 0.9477, Validation Accuracy: 0.9508, Loss: 0.0383
    Epoch   4 Batch   70/538 - Train Accuracy: 0.9660, Validation Accuracy: 0.9600, Loss: 0.0272
    Epoch   4 Batch   80/538 - Train Accuracy: 0.9580, Validation Accuracy: 0.9473, Loss: 0.0299
    Epoch   4 Batch   90/538 - Train Accuracy: 0.9630, Validation Accuracy: 0.9547, Loss: 0.0338
    Epoch   4 Batch  100/538 - Train Accuracy: 0.9730, Validation Accuracy: 0.9512, Loss: 0.0239
    Epoch   4 Batch  110/538 - Train Accuracy: 0.9717, Validation Accuracy: 0.9489, Loss: 0.0299
    Epoch   4 Batch  120/538 - Train Accuracy: 0.9654, Validation Accuracy: 0.9505, Loss: 0.0348
    Epoch   4 Batch  130/538 - Train Accuracy: 0.9615, Validation Accuracy: 0.9494, Loss: 0.0372
    Epoch   4 Batch  140/538 - Train Accuracy: 0.9537, Validation Accuracy: 0.9492, Loss: 0.0419
    Epoch   4 Batch  150/538 - Train Accuracy: 0.9691, Validation Accuracy: 0.9373, Loss: 0.0304
    Epoch   4 Batch  160/538 - Train Accuracy: 0.9596, Validation Accuracy: 0.9411, Loss: 0.0292
    Epoch   4 Batch  170/538 - Train Accuracy: 0.9611, Validation Accuracy: 0.9506, Loss: 0.0343
    Epoch   4 Batch  180/538 - Train Accuracy: 0.9537, Validation Accuracy: 0.9599, Loss: 0.0322
    Epoch   4 Batch  190/538 - Train Accuracy: 0.9541, Validation Accuracy: 0.9627, Loss: 0.0405
    Epoch   4 Batch  200/538 - Train Accuracy: 0.9740, Validation Accuracy: 0.9570, Loss: 0.0222
    Epoch   4 Batch  210/538 - Train Accuracy: 0.9503, Validation Accuracy: 0.9544, Loss: 0.0376
    Epoch   4 Batch  220/538 - Train Accuracy: 0.9550, Validation Accuracy: 0.9553, Loss: 0.0341
    Epoch   4 Batch  230/538 - Train Accuracy: 0.9645, Validation Accuracy: 0.9581, Loss: 0.0288
    Epoch   4 Batch  240/538 - Train Accuracy: 0.9598, Validation Accuracy: 0.9597, Loss: 0.0280
    Epoch   4 Batch  250/538 - Train Accuracy: 0.9582, Validation Accuracy: 0.9311, Loss: 0.0323
    Epoch   4 Batch  260/538 - Train Accuracy: 0.9461, Validation Accuracy: 0.9460, Loss: 0.0333
    Epoch   4 Batch  270/538 - Train Accuracy: 0.9678, Validation Accuracy: 0.9478, Loss: 0.0222
    Epoch   4 Batch  280/538 - Train Accuracy: 0.9762, Validation Accuracy: 0.9467, Loss: 0.0237
    Epoch   4 Batch  290/538 - Train Accuracy: 0.9594, Validation Accuracy: 0.9588, Loss: 0.0270
    Epoch   4 Batch  300/538 - Train Accuracy: 0.9477, Validation Accuracy: 0.9593, Loss: 0.0329
    Epoch   4 Batch  310/538 - Train Accuracy: 0.9793, Validation Accuracy: 0.9524, Loss: 0.0338
    Epoch   4 Batch  320/538 - Train Accuracy: 0.9568, Validation Accuracy: 0.9478, Loss: 0.0312
    Epoch   4 Batch  330/538 - Train Accuracy: 0.9602, Validation Accuracy: 0.9553, Loss: 0.0294
    Epoch   4 Batch  340/538 - Train Accuracy: 0.9518, Validation Accuracy: 0.9451, Loss: 0.0265
    Epoch   4 Batch  350/538 - Train Accuracy: 0.9552, Validation Accuracy: 0.9492, Loss: 0.0356
    Epoch   4 Batch  360/538 - Train Accuracy: 0.9650, Validation Accuracy: 0.9641, Loss: 0.0261
    Epoch   4 Batch  370/538 - Train Accuracy: 0.9619, Validation Accuracy: 0.9466, Loss: 0.0288
    Epoch   4 Batch  380/538 - Train Accuracy: 0.9662, Validation Accuracy: 0.9608, Loss: 0.0262
    Epoch   4 Batch  390/538 - Train Accuracy: 0.9593, Validation Accuracy: 0.9624, Loss: 0.0246
    Epoch   4 Batch  400/538 - Train Accuracy: 0.9695, Validation Accuracy: 0.9599, Loss: 0.0298
    Epoch   4 Batch  410/538 - Train Accuracy: 0.9711, Validation Accuracy: 0.9476, Loss: 0.0309
    Epoch   4 Batch  420/538 - Train Accuracy: 0.9553, Validation Accuracy: 0.9327, Loss: 0.0308
    Epoch   4 Batch  430/538 - Train Accuracy: 0.9559, Validation Accuracy: 0.9462, Loss: 0.0305
    Epoch   4 Batch  440/538 - Train Accuracy: 0.9658, Validation Accuracy: 0.9515, Loss: 0.0326
    Epoch   4 Batch  450/538 - Train Accuracy: 0.9366, Validation Accuracy: 0.9540, Loss: 0.0390
    Epoch   4 Batch  460/538 - Train Accuracy: 0.9622, Validation Accuracy: 0.9398, Loss: 0.0285
    Epoch   4 Batch  470/538 - Train Accuracy: 0.9641, Validation Accuracy: 0.9560, Loss: 0.0280
    Epoch   4 Batch  480/538 - Train Accuracy: 0.9676, Validation Accuracy: 0.9460, Loss: 0.0314
    Epoch   4 Batch  490/538 - Train Accuracy: 0.9630, Validation Accuracy: 0.9604, Loss: 0.0283
    Epoch   4 Batch  500/538 - Train Accuracy: 0.9764, Validation Accuracy: 0.9446, Loss: 0.0216
    Epoch   4 Batch  510/538 - Train Accuracy: 0.9630, Validation Accuracy: 0.9506, Loss: 0.0248
    Epoch   4 Batch  520/538 - Train Accuracy: 0.9559, Validation Accuracy: 0.9469, Loss: 0.0316
    Epoch   4 Batch  530/538 - Train Accuracy: 0.9621, Validation Accuracy: 0.9581, Loss: 0.0362
    Model Trained and Saved
    

### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, we first need to preprocess it.  The function `sentence_to_seq()` is used to preprocess new sentences.

- Converts the sentence to lowercase
- Converts words into ids using `vocab_to_int`
- Converts words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    sentence = sentence.lower()
    words = sentence.split()
    word_id_list = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in words]
    return word_id_list


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed
    

## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         source_sequence_length: [len(translate_sentence)]*batch_size,
                                         keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in translate_logits]))
print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))

```

    INFO:tensorflow:Restoring parameters from checkpoints/dev
    Input
      Word Ids:      [206, 30, 51, 17, 150, 197, 101]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [137, 17, 228, 50, 29, 40, 123, 115, 1]
      French Words: il a vu un vieux camion noir . <EOS>
    

## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus to get better results.  
