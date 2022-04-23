---
title: Predicting Phonemes with BERT
date: 2022-04-23
permalink: /posts/2022/04/predicting-phonemes-with-bert/
tags:
  - Transformer
---

Our team at [Bookbot](https://www.bookbotkids.com/) is currently developing a grapheme-to-phoneme Python package for Bahasa Indonesia. The package is highly inspired by its English counterpart, [g2p](https://github.com/Kyubyong/g2p). A lot of our design and methods are borrowed from that library, most notably the steps to predict phonemes. The English g2p used the following algorithm (c.f. g2p's [README](https://github.com/Kyubyong/g2p#algorithm)):

1. Spells out arabic numbers and some currency symbols. (e.g. $200 -> two hundred dollars) (This is borrowed from Keith Ito's code)
2. Attempts to retrieve the correct pronunciation for heteronyms based on their POS)
3. Looks up The CMU Pronouncing Dictionary for non-homographs.
4. For OOVs, we predict their pronunciations using our neural net model.

Steps 1-3 are particularly easier to develop, granted that we were able to find an online Bahasa Indonesia lexicon from [ipa-dict](https://github.com/open-dict-data/ipa-dict/blob/master/data/ma.txt). Step 4 however, was particularly challenging. Authors of g2p used a recurrent, sequence2sequence [GRU](https://arxiv.org/abs/1409.1259) that takes in graphemes as inputs and outputs phonemes. This approach is particularly useful because we would not need to determine the rules of conversion by hand. The neural net would do the heavy lifting prediction for us for unseen words.

Seeing their success, we attempted a similar approach. That is, we trained a recurrent sequence2sequence [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735) on the aforementioned lexicon, which you can find [here](https://huggingface.co/bookbot/id-g2p-lstm). As expected, the model worked great for words that are relatively simple and words whose sub-words may have been in the training set. It also achieved a validation accuracy of over 97% -- and so we thought it would suffice.

We then converted the model to ONNX for deployment purposes and soon ended up with a working prototype g2p library, using the exact same approach as the English g2p. Upon further playing around, we quickly found an issue with the seq2seq approach. Though it performed well on the held-out validation set, it quickly crumbled when given strikingly different words, for instance names of people or names of a place. On the one hand, this is not surprising given that its training data is relatively small. But we thought we could do better.

First, we realized that phonemes **in the [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) format that our data was in** was not too different from their corresponding graphemes. For instance, here are a few examples:

- `sampingnya` = `sampiŋɲa`
- `tayangan` = `tajaŋan`
- `bepercikan` = `bəpərtʃikan`
- `deduktif` = `deduʔtif`

You may notice that there are simple mapping rules that we could infer by hand. Indeed, we found the following rules to be sufficient

```py
PHONETIC_MAPPING = {
    "ny": "ɲ",
    "ng": "ŋ",
    "c": "tʃ",
    "'": "ʔ",
    "aa": "aʔa",
    "ii": "iʔi",
    "oo": "oʔo",
    "əə": "əʔə",
    "j": "dʒ",
    "y": "j",
    "q": "k"
}

CONSONANTS = "bdfghjklmnprstvwxɲ"

def g2p(text):
    if text.endswith("k"):
        text = text[:-1] + "ʔ"

    for g, p in PHONETIC_MAPPING.items():
        text = text.replace(g, p)

    for c in CONSONANTS:
        text = text.replace(f"k{c}", f"ʔ{c}")

    return text
```

The code is written in Python, with very basic _if-this-then-that_ rules. This approach made a lot of sense, given that changes from a grapheme to an IPA phoneme aren't too drastic, at least in our case. A sequence2sequence model could definitely do the same, but it would probably need a larger and more diverse dataset for training.

But, that doesn't mean that the English g2p approach using a GRU was ineffective! Notice that their phoneme is of the [ARPAbet](https://en.wikipedia.org/wiki/ARPABET) format, which is significantly more complicated than the IPA format we used. Their approach made complete sense because of the change in text domains. This is the same reason why translation tasks are better of using a sequence2sequence neural net over hand-written rules. It would take ages, if not impossible, to code up all rules of translation between 2 languages, but a recurrent model like GRU could automatically learn this "hidden translation rule" if there was one.

# A problem with the letter E

But there was a huge issue with the rule-based approach we took. That is, there are 3 ways to pronounce the letter `e` in Indonesian, according to [KBBI](https://ivanlanin.github.io/puebi/huruf/huruf-vokal/). The lexicon that we used further limited the pronunciation to only two ways: a closed-mid front unrounded vowel `e` or a mid central vowel `ə`. For example, the word `bebek` (meaning: duck) has the phoneme `bebek`, while the word `delapan` (meaning: eight) has the phoneme `dəlapan`. Sometimes, a word might have >1 `e`'s pronounced in both ways, like the word `mereka` (meaning: they) that is pronounced as `məreka`. You can hear how they sound through the Google Translate TTS [here](https://translate.google.com/?sl=id&tl=en&text=bebek&op=translate), [here](https://translate.google.com/?sl=id&tl=en&text=delapan&op=translate), and [here](https://translate.google.com/?sl=id&tl=en&text=mereka&op=translate).

To the best of our knowledge, there isn't a linguistic rule to determine exactly how a particular `e` should sound like. KBBI might have phonetic assistance for this purpose, particularly homographs. Non-homographs, however, do not have phonetic assistance. I personally think that this is a huge problem, especially for new learners of the language. Native speakers like me would find this distinction of `e`'s as natural, but I can't imagine being in the shoes of someone learning the language.

To be fair, the Indonesian language isn't like the English language where there are "native speakers" to whom we can consult. The Indonesian language is a lingua franca, a standardized version of Malay, and was largely influenced by Dutch and tons of other regional languages such as Javanese, Sundanese, etc. There might not necessarily be a definitive "correct" way to pronounce the letter `e` of a given word, because in order to do so, we need to consult the origin of the word. Furthermore, different regions of Indonesia may pronounce the same word differently, due to their dialect. You can read more about this [here](https://id.quora.com/Mengapa-terdapat-perbedaan-pelafalan-huruf-E-dalam-beberapa-kata-yang-berbahasa-Indonesia-Contohnya-bendera-dan-benderang/answer/Benny-Lin) and here [here](https://id.quora.com/Mengapa-terdapat-perbedaan-pelafalan-huruf-E-dalam-beberapa-kata-yang-berbahasa-Indonesia-Contohnya-bendera-dan-benderang/answer/Gladhys-Elliona-Syahutari). Both discussions are in Indonesian, but Google Translate should do the job.

In any case, our g2p package needs a way to distinguish `e`'s from `ə`'s. Once that distinction has been made, we can simply pass it to the hand-written g2p algorithm that does the rest of the job.

# Formulating the Problem

At first, we thought a sequence2sequence can do the job just fine. We can simply train on pairs of data like:

- `bebek` & `bebek`
- `delapan` & `dəlapan`
- `mereka` & `məreka`

and then simply pass their output to the hand-written g2p rule. But after more thinking, we recalled the pitfalls of this method and thought that it would suffer from the same issues. Bad OOV performance, incorrect output length, etc. And so we re-formulated the problem differently.

Instead of treating the phonetic prediction as a generation problem, why not treat it as a de-masking problem? That is, instead of training an autoregressive model like an LSTM, why not train an autoencoder model like [BERT](https://arxiv.org/abs/1810.04805) instead?

Normally, a BERT model is trained as a word-level masked language model; think fill in the blanks problem. Given the context:

```
The weather is good today, the ___ is bright and blue.
```

or

```
Have a ____ and relax.
```

You can probably infer what those blanks should be. And that is exactly how BERT is trained. It sees the neighbors of the masked (emptied) word, and makes a prediction based on them. Realizing this, I saw a very intruiging possibility to implement the same mechanics for our problem with the letter `e`. That is, frame the problem as:

- Context: `b _ b _ k`, Output: `b e b e k`
- Context: `d _ l a p a n`, Output: `d ə l a p a n`
- Context: `m _ r _ k a`, Output: `m ə r e k a`

and so on. The hope is that, given the neighbouring letters, the BERT model will be able to infer the right phoneme of `e` to use.

Per my research, I have not found someone else using the same approach. I don't know if the idea is merely bad on paper, so I gave it a try because, why not?

# Code

## Dataset

This is the training dataset that I ended up with. But recall, we need to mask out the `e`'s later and let the model predict the suitable phonetic `e`. Again, this dataset originates from the [ipa-dict](https://github.com/open-dict-data/ipa-dict/blob/master/data/ma.txt) which we pre-processed and modified. You can find our version [here](https://huggingface.co/datasets/bookbot/id_word2phoneme).

|       | word                | target              |
| ----- | ------------------- | ------------------- |
| 0     | - - n y a           | - - n y a           |
| 1     | - a n d a           | - a n d a           |
| 2     | - b a u r           | - b a u r           |
| 3     | - b e l a s         | - b ə l a s         |
| 4     | - c o m p e n g     | - c o m p e n g     |
| ...   | ...                 | ...                 |
| 27547 | z o h o r           | z o h o r           |
| 27548 | z o n a             | z o n a             |
| 27549 | z u h u r           | z u h u r           |
| 27550 | z u l k a r n a i n | z u l k a r n a i n |
| 27551 | z u r i a t         | z u r i a t         |

## Character-Level Masked Language Model

Now, I have never written a BERT Masked Language Model from scratch, so I followed a very nice guide from [Keras](https://keras.io/examples/nlp/masked_language_modeling/), written by [Ankur Singh](https://twitter.com/ankur310794). It's very clear and easily customizable to our use case, so I went with it.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
```

```python
@dataclass
class Config:
    MAX_LEN = 32
    BATCH_SIZE = 128
    LR = 0.001
    VOCAB_SIZE = 32
    EMBED_DIM = 128
    NUM_HEAD = 8
    FF_DIM = 128
    NUM_LAYERS = 2

config = Config()
```

### Tokenization and Preprocessing

The tutorial used a Keras `TextVectorization` layer for tokenization purposes, which I also find to be easy to use and customize. The only change I made was simplifying the text standarization function.

```python
def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=lambda input_data: tf.strings.lower(input_data),
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    vocab = vectorize_layer.get_vocabulary()

    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer

vectorize_layer = get_vectorize_layer(
    df.target.values.tolist(),
    config.VOCAB_SIZE,
    config.MAX_LEN,
    special_tokens=["[mask]"],
)
```

This is where most of the changes were made. First, instead of masking characters at random, only a "hard-mask" was applied on both `e` and `ə` tokens, completely masking them out in every text. This meant that the 15% BERT masking, 90%/10% random masking, as well as the 10% random swaps were all removed. I found that masking other characters which are not `e`'s gave worse performance. I suspect that this just made the problem even harder for the model to learn since there is very minimal context.

```python
# Get mask token id for masked language model
mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]
e1_token_id = vectorize_layer(["e"]).numpy()[0][0]
e2_token_id = vectorize_layer(["ə"]).numpy()[0][0]

def encode(texts):
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()

def get_masked_input_and_labels(encoded_texts):
    # BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Force mask e's
    inp_mask[encoded_texts == e1_token_id] = True
    inp_mask[encoded_texts == e2_token_id] = True
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    encoded_texts_masked[inp_mask] = mask_token_id
    # note: we don't randomly change chars and apply all masks

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights
```

Here's an example of an input, label, and weights array, respectively. Notice that at the index of the letter `e`, the input is masked and has the mask token id of `30`, with the target token id of `18` and `4`, corresponding to `e` and `ə`, respectively. Also notice that the weights default to `0` for unmasked tokens and `1` for masked tokens. This is to facilitate training. Recall that the model will only be "graded" by its performance on the blanks.

```python
get_masked_input_and_labels(encode("m e r d ə k a"))
```

    (array([ 8, 30,  6, 16, 30,  7,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
     array([ 8, 18,  6, 16,  4,  7,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
     array([0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

```python
# Prepare data for masked language model
x_all = encode(df.target.values)
x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(x_all)

mlm_ds = tf.data.Dataset.from_tensor_slices(
    (x_masked_train, y_masked_labels, sample_weights)
)
mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)
```

### BERT

There's really no difference between the code written in the Keras guide with the one I have here. I'll just note how elegant Keras code is for a model like BERT. But in any case, this model is exactly the same as if we were to train a word-level masked language model. This time, the input tokens are just characters instead of words. Same old objective, same architecture, and so on.

```python
def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = tf.keras.metrics.Mean(name="loss")


class MaskedLanguageModel(tf.keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker]


def create_masked_language_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)

    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = MaskedLanguageModel(inputs, mlm_output, name="masked_bert_model")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model
```

```python
id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}

bert_masked_model = create_masked_language_bert_model()
bert_masked_model.summary()
```

    Model: "masked_bert_model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_1 (InputLayer)           [(None, 32)]         0           []

     word_embedding (Embedding)     (None, 32, 128)      4096        ['input_1[0][0]']

     tf.__operators__.add (TFOpLamb  (None, 32, 128)     0           ['word_embedding[0][0]']
     da)

     encoder_0/multiheadattention (  (None, 32, 128)     66048       ['tf.__operators__.add[0][0]',
     MultiHeadAttention)                                              'tf.__operators__.add[0][0]',
                                                                      'tf.__operators__.add[0][0]']

     encoder_0/att_dropout (Dropout  (None, 32, 128)     0           ['encoder_0/multiheadattention[0]
     )                                                               [0]']

     tf.__operators__.add_1 (TFOpLa  (None, 32, 128)     0           ['tf.__operators__.add[0][0]',
     mbda)                                                            'encoder_0/att_dropout[0][0]']

     encoder_0/att_layernormalizati  (None, 32, 128)     256         ['tf.__operators__.add_1[0][0]']
     on (LayerNormalization)

     encoder_0/ffn (Sequential)     (None, 32, 128)      33024       ['encoder_0/att_layernormalizatio
                                                                     n[0][0]']

     encoder_0/ffn_dropout (Dropout  (None, 32, 128)     0           ['encoder_0/ffn[0][0]']
     )

     tf.__operators__.add_2 (TFOpLa  (None, 32, 128)     0           ['encoder_0/att_layernormalizatio
     mbda)                                                           n[0][0]',
                                                                      'encoder_0/ffn_dropout[0][0]']

     encoder_0/ffn_layernormalizati  (None, 32, 128)     256         ['tf.__operators__.add_2[0][0]']
     on (LayerNormalization)

     encoder_1/multiheadattention (  (None, 32, 128)     66048       ['encoder_0/ffn_layernormalizatio
     MultiHeadAttention)                                             n[0][0]',
                                                                      'encoder_0/ffn_layernormalizatio
                                                                     n[0][0]',
                                                                      'encoder_0/ffn_layernormalizatio
                                                                     n[0][0]']

     encoder_1/att_dropout (Dropout  (None, 32, 128)     0           ['encoder_1/multiheadattention[0]
     )                                                               [0]']

     tf.__operators__.add_3 (TFOpLa  (None, 32, 128)     0           ['encoder_0/ffn_layernormalizatio
     mbda)                                                           n[0][0]',
                                                                      'encoder_1/att_dropout[0][0]']

     encoder_1/att_layernormalizati  (None, 32, 128)     256         ['tf.__operators__.add_3[0][0]']
     on (LayerNormalization)

     encoder_1/ffn (Sequential)     (None, 32, 128)      33024       ['encoder_1/att_layernormalizatio
                                                                     n[0][0]']

     encoder_1/ffn_dropout (Dropout  (None, 32, 128)     0           ['encoder_1/ffn[0][0]']
     )

     tf.__operators__.add_4 (TFOpLa  (None, 32, 128)     0           ['encoder_1/att_layernormalizatio
     mbda)                                                           n[0][0]',
                                                                      'encoder_1/ffn_dropout[0][0]']

     encoder_1/ffn_layernormalizati  (None, 32, 128)     256         ['tf.__operators__.add_4[0][0]']
     on (LayerNormalization)

     mlm_cls (Dense)                (None, 32, 32)       4128        ['encoder_1/ffn_layernormalizatio
                                                                     n[0][0]']

    ==================================================================================================
    Total params: 207,392
    Trainable params: 207,392
    Non-trainable params: 0
    __________________________________________________________________________________________________

### Train!

What's left is just for us to call `.fit()`, because this is Keras. The Keras guide used the [Adam optimizer](https://arxiv.org/abs/1412.6980), which generally works well for language models.

```python
bert_masked_model.fit(
    mlm_ds, epochs=100, callbacks=[keras.callbacks.TensorBoard(log_dir="./logs")]
)

bert_masked_model.save("bert_mlm.h5")
```

    Epoch 1/100
    216/216 [==============================] - 8s 13ms/step - loss: 0.4276
    Epoch 2/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.3865
    Epoch 3/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.3320
    Epoch 4/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.3048
    Epoch 5/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2887
    Epoch 6/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2870
    Epoch 7/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2827
    Epoch 8/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2795
    Epoch 9/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2939
    Epoch 10/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2751
    Epoch 11/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2743
    Epoch 12/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2678
    Epoch 13/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2671
    Epoch 14/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2609
    Epoch 15/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2619
    Epoch 16/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2681
    Epoch 17/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2689
    Epoch 18/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2582
    Epoch 19/100
    216/216 [==============================] - 4s 16ms/step - loss: 0.2526
    Epoch 20/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2559
    Epoch 21/100
    216/216 [==============================] - 3s 14ms/step - loss: 0.2506
    Epoch 22/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2548
    Epoch 23/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2584
    Epoch 24/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2502
    Epoch 25/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2484
    Epoch 26/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2448
    Epoch 27/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2502
    Epoch 28/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2471
    Epoch 29/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2471
    Epoch 30/100
    216/216 [==============================] - 4s 20ms/step - loss: 0.2422
    Epoch 31/100
    216/216 [==============================] - 5s 22ms/step - loss: 0.2412
    Epoch 32/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2398
    Epoch 33/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2500
    Epoch 34/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2445
    Epoch 35/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2407
    Epoch 36/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2376
    Epoch 37/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2351
    Epoch 38/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2363
    Epoch 39/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2377
    Epoch 40/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2351
    Epoch 41/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2467
    Epoch 42/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2408
    Epoch 43/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2332
    Epoch 44/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2355
    Epoch 45/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2371
    Epoch 46/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2353
    Epoch 47/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2293
    Epoch 48/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2270
    Epoch 49/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2258
    Epoch 50/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2255
    Epoch 51/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2240
    Epoch 52/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2309
    Epoch 53/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2336
    Epoch 54/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2297
    Epoch 55/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2279
    Epoch 56/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2245
    Epoch 57/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2239
    Epoch 58/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2225
    Epoch 59/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2237
    Epoch 60/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2213
    Epoch 61/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2210
    Epoch 62/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2186
    Epoch 63/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2187
    Epoch 64/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2191
    Epoch 65/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2165
    Epoch 66/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2172
    Epoch 67/100
    216/216 [==============================] - 5s 23ms/step - loss: 0.2182
    Epoch 68/100
    216/216 [==============================] - 4s 20ms/step - loss: 0.2143
    Epoch 69/100
    216/216 [==============================] - 5s 23ms/step - loss: 0.2171
    Epoch 70/100
    216/216 [==============================] - 4s 19ms/step - loss: 0.2096
    Epoch 71/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2122
    Epoch 72/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2169
    Epoch 73/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2134
    Epoch 74/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2117
    Epoch 75/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2094
    Epoch 76/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2123
    Epoch 77/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2134
    Epoch 78/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2117
    Epoch 79/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2064
    Epoch 80/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2111
    Epoch 81/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2130
    Epoch 82/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2089
    Epoch 83/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2063
    Epoch 84/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2042
    Epoch 85/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2032
    Epoch 86/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2071
    Epoch 87/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2062
    Epoch 88/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.1999
    Epoch 89/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2021
    Epoch 90/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2019
    Epoch 91/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2056
    Epoch 92/100
    216/216 [==============================] - 4s 16ms/step - loss: 0.2062
    Epoch 93/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2006
    Epoch 94/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.2034
    Epoch 95/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2003
    Epoch 96/100
    216/216 [==============================] - 3s 12ms/step - loss: 0.2005
    Epoch 97/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.1970
    Epoch 98/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.1951
    Epoch 99/100
    216/216 [==============================] - 3s 13ms/step - loss: 0.1960
    Epoch 100/100
    216/216 [==============================] - 4s 20ms/step - loss: 0.1991

### Inference

It's also quite simple to perform inference once the model finished training. We first need to load the model and its weights.

```python
# Load pretrained bert model
mlm_model = keras.models.load_model(
    "bert_mlm.h5", custom_objects={"MaskedLanguageModel": MaskedLanguageModel}
)
```

And then write up an inference function which we can reuse later. The way it works is also quite clear. Tokenize the input tokens as integers, while masking the `e`'s to be predicted. Then, pad the inputs to the maximum sequence length (in our case 32) and feed the input array to the BERT model. Decoding the output involves us finding the locations of those masked inputs, finding the most probable guess, and replacing the masked tokens with that prediction. Finally, we join the tokens once in they are all assembled.

```python
def inference(sequence):
    sequence = " ".join([c if c != "e" else "[mask]" for c in sequence])
    tokens = [token2id[c] for c in sequence.split()]
    pad = [token2id[""] for _ in range(config.MAX_LEN - len(tokens))]

    tokens = tokens + pad
    input_ids = tf.convert_to_tensor(np.array([tokens]))
    prediction = mlm_model.predict(input_ids)

    # find masked idx token
    masked_index = np.where(input_ids == mask_token_id)
    masked_index = masked_index[1]

    # get prediction at those masked index only
    mask_prediction = prediction[0][masked_index]
    predicted_ids = np.argmax(mask_prediction, axis=1)

    # replace mask with predicted token
    for i, idx in enumerate(masked_index):
        tokens[idx] = predicted_ids[i]

    return "".join([id2token[t] for t in tokens if t != 0])
```

```python
inference("menyebabkannya")
```

    'mənyəbabkannya'

Not forgetting to apply the hand-written g2p rules that we came up with.

```python
g2p(inference("menyebabkannya"))
```

    'məɲəbabkanɲa'

And thus we are done.

In practice, I would convert the Keras model over to ONNX so that I can run the static model with only NumPy as a dependency instead of TensorFlow/Keras. But it's really up to your use case.

# Conclusion

This little weekend experiment of mine is pretty much just a proof of concept, certainly with room for improvements. But at least, I'm happy that it worked better than the LSTM. It's much more controllable and won't be too shabby of a guess for OOV words.

This will be available once the g2p package we're developing becomes open source. Hopefully it is by the time that this blog post becomes live. Otherwise, we're still working on it :)
