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
    """Pre-layer norm, otherwise standard."""
    # MHA block
    mha_pre_ln = layers.LayerNormalization()(inputs)
    mha_out = layers.MultiHeadAttention(
        key_dim=width // heads, num_heads=heads, dropout=dropout
    )(mha_pre_ln, mha_pre_ln, attention_mask=attention_mask)
    mha_dropout = layers.Dropout(dropout)(mha_out)
    mha_add = layers.add([mha_dropout, inputs])

    # FF block
    ff_pre_ln = layers.LayerNormalization()(mha_add)
    ff_dense_1_out = layers.Dense(width, activation="relu")(ff_pre_ln)
    ff_dense_2_out = layers.Dense(width)(ff_dense_1_out)
    ff_dropout = layers.Dropout(dropout)(ff_dense_2_out)
    ff_add = layers.add([ff_dropout, mha_add])

    return ff_add


def decoder_transformer_classifier(
    seq_len: int,
    vocabulary_size: int,
    n_classes: int,
    n_layers: int,
    width: int,
    heads: int,
    dropout: Optional[float] = None,
) -> tf.keras.Model:
    """Implements a standard transformer decoder except with pre-layer
    normalization as described in Xiong et al. (2020)."""
    attention_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    inputs = tf.keras.Input((seq_len,))
    x = layers.Embedding(vocabulary_size, width)(inputs)
    for _ in range(n_layers):
        x = transformer_layer(x, width, heads, attention_mask, dropout)
    x_ln = layers.LayerNormalization()(x)
    logits = layers.Dense(n_classes, name="to_logits")(x_ln[..., -1, :])
    return tf.keras.Model(inputs=[inputs], outputs=[logits])
