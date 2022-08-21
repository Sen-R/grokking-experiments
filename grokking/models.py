from typing import Optional
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers  # type: ignore


def transformer_layer(
    inputs,
    width: int,
    heads: int,
    attention_mask: tf.Tensor,
    dropout: Optional[float] = None,
):
    # MHA block
    mha_out = layers.MultiHeadAttention(
        key_dim=width // heads, num_heads=heads, dropout=dropout
    )(inputs, inputs, attention_mask=attention_mask)
    mha_dropout = layers.Dropout(dropout)(mha_out)
    mha_add = layers.add([mha_dropout, inputs])
    mha_ln = layers.LayerNormalization()(mha_add)

    # FF block
    dense_1_out = layers.Dense(width, activation="relu")(mha_ln)
    dense_2_out = layers.Dense(width)(dense_1_out)
    ff_dropout = layers.Dropout(dropout)(dense_2_out)
    ff_add = layers.add([ff_dropout, mha_ln])
    ff_ln = layers.LayerNormalization()(ff_add)

    return ff_ln


def decoder_transformer_classifier(
    seq_len: int,
    vocabulary_size: int,
    n_classes: int,
    n_layers: int,
    width: int,
    heads: int,
    dropout: Optional[float] = None,
) -> tf.keras.Model:
    attention_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    inputs = tf.keras.Input((seq_len,))
    x = layers.Embedding(vocabulary_size, width)(inputs)
    for _ in range(n_layers):
        x = transformer_layer(x, width, heads, attention_mask, dropout)
    logits = layers.Dense(n_classes, name="to_logits")(x[..., -1, :])
    return tf.keras.Model(inputs=[inputs], outputs=[logits])
