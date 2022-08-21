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
    ln_1_out = layers.LayerNormalization()(inputs)
    mha_out = layers.MultiHeadAttention(
        key_dim=width // heads, num_heads=heads, dropout=dropout
    )(ln_1_out, ln_1_out, attention_mask=attention_mask)
    mha_post_dropout = layers.Dropout(dropout)(mha_out)
    mha_block_out = layers.add([mha_post_dropout, inputs])  # resid connection

    # FF block
    ln_2_out = layers.LayerNormalization()(mha_block_out)
    dense_1_out = layers.Dense(width, activation="relu")(ln_2_out)
    dense_1_post_dropout = layers.Dropout(dropout)(dense_1_out)
    dense_2_out = layers.Dense(width)(dense_1_post_dropout)
    ff_block_out = layers.add([dense_2_out, mha_block_out])  # resid conn

    return ff_block_out


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
    logits = layers.Dense(n_classes)(x[..., -1, :])
    return tf.keras.Model(inputs=[inputs], outputs=[logits])
