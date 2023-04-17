# machine_translation_with_SeqtoSeq_Attention_model
Machine Translation English to Hindi with Deep Learning Seq-to-Seq Encoder-Decoder Model with Attention
This project aims to develop a machine translation system that can translate English sentences to Hindi sentences using a deep learning sequence-to-sequence encoder-decoder model with attention mechanism. The attention mechanism helps the model to focus on specific parts of the input sentence while generating the output sentence, which improves the translation quality.

### Requirements
This project requires the following libraries:\
Python 3.x \
TensorFlow 2.x\
NumPy\
NLTK\
Pandas\
sklearn\
unicodedata\

These libraries can be easily installed using pip or conda package manager.

### Dataset
The project uses the parallel corpus dataset of English-Hindi sentence pairs, which can be downloaded from the following link: [dataset](https://www.kaggle.com/aiswaryaramachandran/hindienglish-corpora).

The dataset contains 1,56,327 sentence pairs, with each sentence in English and Hindi.

### Model Architecture
The model architecture used in this project is a sequence-to-sequence encoder-decoder model with an attention mechanism. The encoder is a bidirectional LSTM network, which encodes the input sentence into a fixed-length vector. The decoder is another LSTM network, which decodes the encoded vector into the output sentence. The attention mechanism is used to weigh the importance of each input token at each decoding step, improving the translation quality.

### Usage
To train the model, run the above file.The trained model will be saved in the model directory, and the translated output can be stored in the 'output' directory.
#### The below python code for encoder :
```
class Encoder(tf.keras.Model):
    def __init__(self, source_vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = layers.Embedding(source_vocab_size, 
                                          embedding_dim,
                                          name='encoder_embedding_layer')
        self.lstm = layers.LSTM(hidden_dim, 
                                return_sequences=True, 
                                return_state=True,
                                name='encoder_lstm')
    def call(self, input):
        embeddings = self.embedding(input)
        output_seq, state_h, state_c = self.lstm(embeddings)
        return output_seq, state_h, state_c
```
#### The below python code for decoder :
```
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.embedding_layer = layers.Embedding(vocab_size,
                                                embedding_dim,
                                                name='decoder_embedding_layer')

        self.lstm = layers.LSTM(hidden_dim,
                                return_sequences=True,
                                return_state=True,
                                name='decoder_lstm')

        self.attention = LuongAttention(hidden_dim)

        self.w = tf.keras.layers.Dense(hidden_dim, activation='tanh', name='attended_outputs_dense')

        self.dense = layers.Dense(vocab_size, name='decoder_dense')


    def call(self, inputs):
        decoder_input, encoder_output_seq, lstm_state = inputs
        embeddings = self.embedding_layer(decoder_input)

        decoder_output, state_h, state_c = self.lstm(embeddings, initial_state=lstm_state)

        weights, context = self.attention([encoder_output_seq, decoder_output])

        decoder_output_with_attention = self.w(tf.concat(
            [tf.squeeze(context, 1), tf.squeeze(decoder_output, 1)], -1))

        logits = self.dense(decoder_output_with_attention)

        return logits, state_h, state_c, weights
```
### Results
![image](https://user-images.githubusercontent.com/130960032/232437537-1eb9e2a5-b079-422e-8318-f8083411860a.png)


### Acknowledgments
This project is based on the research paper "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau et al. (https://arxiv.org/pdf/1409.0473.pdf). The implementation of the attention mechanism is inspired by the TensorFlow tutorial on Neural Machine Translation with Attention (https://www.tensorflow.org/tutorials/text/nmt_with_attention).

### Conclusion
This project demonstrates the use of deep learning sequence-to-sequence encoder-decoder models with attention mechanisms for machine translation tasks. The model achieves decent translation quality on the English-Hindi corpus dataset and can be further improved with more training data and fine-tuning of hyperparameters.
