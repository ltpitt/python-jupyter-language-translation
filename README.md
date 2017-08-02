
# Language Translation
In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.


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
Play around with view_sentence_range to view different parts of the data.


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
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


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
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


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

    TensorFlow Version: 1.3.0-rc1
    

    C:\Users\DNastri\AppData\Local\Continuum\Anaconda3\envs\language_best\lib\site-packages\ipykernel_launcher.py:15: UserWarning: No GPU found. Please use a GPU to train your neural network.
      from ipykernel import kernelapp as app
    

## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoder_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
- Target sequence length placeholder named "target_sequence_length" with rank 1
- Max target sequence length tensor named "max_target_len" getting its value from applying tf.reduce_max on the target_sequence_length placeholder. Rank 0.
- Source sequence length placeholder named "source_sequence_length" with rank 1

Return the placeholders in the following the tuple (input, targets, learning rate, keep probability, target sequence length, max target sequence length, source sequence length)


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

    ERROR:tensorflow:==================================
    Object was never used (type <class 'tensorflow.python.framework.ops.Operation'>):
    <tf.Operation 'assert_rank_2/Assert/Assert' type=Assert>
    If you want to mark it as used call its "mark_used()" method.
    It was originally created here:
    ['File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\runpy.py", line 193, in _run_module_as_main\n    "__main__", mod_spec)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\runpy.py", line 85, in _run_code\n    exec(code, run_globals)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel_launcher.py", line 16, in <module>\n    app.launch_new_instance()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\traitlets\\config\\application.py", line 658, in launch_instance\n    app.start()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelapp.py", line 477, in start\n    ioloop.IOLoop.instance().start()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\ioloop.py", line 177, in start\n    super(ZMQIOLoop, self).start()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tornado\\ioloop.py", line 888, in start\n    handler_func(fd_obj, events)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tornado\\stack_context.py", line 277, in null_wrapper\n    return fn(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\zmqstream.py", line 440, in _handle_events\n    self._handle_recv()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\zmqstream.py", line 472, in _handle_recv\n    self._run_callback(callback, msg)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\zmqstream.py", line 414, in _run_callback\n    callback(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tornado\\stack_context.py", line 277, in null_wrapper\n    return fn(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelbase.py", line 283, in dispatcher\n    return self.dispatch_shell(stream, msg)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelbase.py", line 235, in dispatch_shell\n    handler(stream, idents, msg)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelbase.py", line 399, in execute_request\n    user_expressions, allow_stdin)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\ipkernel.py", line 196, in do_execute\n    res = shell.run_cell(code, store_history=store_history, silent=silent)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\zmqshell.py", line 533, in run_cell\n    return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\IPython\\core\\interactiveshell.py", line 2698, in run_cell\n    interactivity=interactivity, compiler=compiler, result=result)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\IPython\\core\\interactiveshell.py", line 2808, in run_ast_nodes\n    if self.run_code(code, result):', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\IPython\\core\\interactiveshell.py", line 2862, in run_code\n    exec(code_obj, self.user_global_ns, self.user_ns)', 'File "<ipython-input-17-4b8937721a82>", line 19, in <module>\n    tests.test_model_inputs(model_inputs)', 'File "C:\\Users\\DNastri\\DeepLearning\\python-jupyter-language-translation\\problem_unittests.py", line 106, in test_model_inputs\n    assert tf.assert_rank(lr, 0, message=\'Learning Rate has wrong rank\')', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\ops\\check_ops.py", line 617, in assert_rank\n    dynamic_condition, data, summarize)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\ops\\check_ops.py", line 571, in _assert_rank_condition\n    return control_flow_ops.Assert(condition, data, summarize=summarize)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\util\\tf_should_use.py", line 175, in wrapped\n    return _add_should_use_warning(fn(*args, **kwargs))', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\util\\tf_should_use.py", line 144, in _add_should_use_warning\n    wrapped = TFShouldUseWarningWrapper(x)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\util\\tf_should_use.py", line 101, in __init__\n    stack = [s.strip() for s in traceback.format_stack()]']
    ==================================
    ERROR:tensorflow:==================================
    Object was never used (type <class 'tensorflow.python.framework.ops.Operation'>):
    <tf.Operation 'assert_rank_3/Assert/Assert' type=Assert>
    If you want to mark it as used call its "mark_used()" method.
    It was originally created here:
    ['File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\runpy.py", line 193, in _run_module_as_main\n    "__main__", mod_spec)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\runpy.py", line 85, in _run_code\n    exec(code, run_globals)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel_launcher.py", line 16, in <module>\n    app.launch_new_instance()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\traitlets\\config\\application.py", line 658, in launch_instance\n    app.start()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelapp.py", line 477, in start\n    ioloop.IOLoop.instance().start()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\ioloop.py", line 177, in start\n    super(ZMQIOLoop, self).start()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tornado\\ioloop.py", line 888, in start\n    handler_func(fd_obj, events)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tornado\\stack_context.py", line 277, in null_wrapper\n    return fn(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\zmqstream.py", line 440, in _handle_events\n    self._handle_recv()', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\zmqstream.py", line 472, in _handle_recv\n    self._run_callback(callback, msg)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\zmq\\eventloop\\zmqstream.py", line 414, in _run_callback\n    callback(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tornado\\stack_context.py", line 277, in null_wrapper\n    return fn(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelbase.py", line 283, in dispatcher\n    return self.dispatch_shell(stream, msg)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelbase.py", line 235, in dispatch_shell\n    handler(stream, idents, msg)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\kernelbase.py", line 399, in execute_request\n    user_expressions, allow_stdin)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\ipkernel.py", line 196, in do_execute\n    res = shell.run_cell(code, store_history=store_history, silent=silent)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\ipykernel\\zmqshell.py", line 533, in run_cell\n    return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\IPython\\core\\interactiveshell.py", line 2698, in run_cell\n    interactivity=interactivity, compiler=compiler, result=result)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\IPython\\core\\interactiveshell.py", line 2808, in run_ast_nodes\n    if self.run_code(code, result):', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\IPython\\core\\interactiveshell.py", line 2862, in run_code\n    exec(code_obj, self.user_global_ns, self.user_ns)', 'File "<ipython-input-17-4b8937721a82>", line 19, in <module>\n    tests.test_model_inputs(model_inputs)', 'File "C:\\Users\\DNastri\\DeepLearning\\python-jupyter-language-translation\\problem_unittests.py", line 107, in test_model_inputs\n    assert tf.assert_rank(keep_prob, 0, message=\'Keep Probability has wrong rank\')', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\ops\\check_ops.py", line 617, in assert_rank\n    dynamic_condition, data, summarize)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\ops\\check_ops.py", line 571, in _assert_rank_condition\n    return control_flow_ops.Assert(condition, data, summarize=summarize)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\util\\tf_should_use.py", line 175, in wrapped\n    return _add_should_use_warning(fn(*args, **kwargs))', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\util\\tf_should_use.py", line 144, in _add_should_use_warning\n    wrapped = TFShouldUseWarningWrapper(x)', 'File "C:\\Users\\DNastri\\AppData\\Local\\Continuum\\Anaconda3\\envs\\language_best\\lib\\site-packages\\tensorflow\\python\\util\\tf_should_use.py", line 101, in __init__\n    stack = [s.strip() for s in traceback.format_stack()]']
    ==================================
    Tests Passed
    

### Process Decoder Input
Implement `process_decoder_input` by removing the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


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
Implement `encoding_layer()` to create a Encoder RNN layer:
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
    cell = tf.contrib.rnn.MultiRNNCell([ tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers) ])
    RNN_output, RNN_state = tf.nn.dynamic_rnn(cell, encoder_inputs, sequence_length=source_sequence_length, dtype=tf.float32)
    return RNN_output, RNN_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed
    

### Decoding - Training
Create a training decoding layer:
* Create a [`tf.contrib.seq2seq.TrainingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper) 
* Create a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtain the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


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
    
    drop = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    train_dec_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(drop, train_dec_fn, dec_embed_input,
                                                              sequence_length, scope=decoding_scope)
    
    return output_fn(train_pred)



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-10-c46624ef59fe> in <module>()
         27 DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
         28 """
    ---> 29 tests.test_decoding_layer_train(decoding_layer_train)
    

    ~\DeepLearning\python-jupyter-language-translation\problem_unittests.py in test_decoding_layer_train(decoding_layer_train)
        340                                         max_target_sequence_length,
        341                                         output_layer,
    --> 342                                         keep_prob)
        343 
        344             # encoder_state, dec_cell, dec_embed_input, sequence_length,
    

    <ipython-input-10-c46624ef59fe> in decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_summary_length, output_layer, keep_prob)
         16 
         17     drop = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    ---> 18     train_dec_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
         19     train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(drop, train_dec_fn, dec_embed_input,
         20                                                               sequence_length, scope=decoding_scope)
    

    AttributeError: module 'tensorflow.contrib.seq2seq' has no attribute 'simple_decoder_fn_train'


### Decoding - Inference
Create inference decoder:
* Create a [`tf.contrib.seq2seq.GreedyEmbeddingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper)
* Create a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtain the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


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
    start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')

    # Helper for the inference process.
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                start_tokens,
                                                                target_letter_to_int['<EOS>'])

    # Basic decoder
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        enc_state,
                                                        output_layer)
        
    # Perform dynamic decoding using the decoder
    inference_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=max_target_sequence_length)



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-3c8931425077> in <module>()
         40 DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
         41 """
    ---> 42 tests.test_decoding_layer_infer(decoding_layer_infer)
    

    ~\DeepLearning\python-jupyter-language-translation\problem_unittests.py in test_decoding_layer_infer(decoding_layer_infer)
        399                                                         output_layer,
        400                                                         batch_size,
    --> 401                                                         keep_prob)
        402 
        403             # encoder_state, dec_cell, dec_embeddings, 10, 20,
    

    <ipython-input-11-3c8931425077> in decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob)
         17     :return: BasicDecoderOutput containing inference logits and sample_id
         18     """
    ---> 19     start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')
         20 
         21     # Helper for the inference process.
    

    NameError: name 'target_letter_to_int' is not defined


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

* Embed the target sequences
* Construct the decoder LSTM cell (just like you constructed the encoder cell above)
* Create an output layer to map the outputs of the decoder to the elements of our vocabulary
* Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)` function to get the training logits.
* Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
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
    # TODO: Implement Function
    return None, None



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

### Build the Neural Network
Apply the functions you implemented above to:

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
    # TODO: Implement Function
    return None, None


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

## Neural Network Training
### Hyperparameters
Tune the following parameters:

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
epochs = None
# Batch Size
batch_size = None
# RNN Size
rnn_size = None
# Number of Layers
num_layers = None
# Embedding Size
encoding_embedding_size = None
decoding_embedding_size = None
# Learning Rate
learning_rate = None
# Dropout Keep Probability
keep_probability = None
display_step = None
```

### Build the Graph
Build the graph using the neural network you implemented.


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
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


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
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    return None


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

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

## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
