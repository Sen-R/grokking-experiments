from typing import Optional, Sequence, Dict, Any, Callable
import numpy as np
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
    n_input_tokens: int,
    n_output_tokens: int,
    n_layers: int,
    width: int,
    heads: int,
    dropout: Optional[float] = None,
) -> tf.keras.Model:
    """Implements a standard transformer decoder except with pre-layer
    normalization as described in Xiong et al. (2020)."""
    attention_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    inputs = tf.keras.Input((seq_len,))
    x = layers.Embedding(n_input_tokens, width)(inputs)
    for _ in range(n_layers):
        x = transformer_layer(x, width, heads, attention_mask, dropout)
    x_ln = layers.LayerNormalization()(x)
    logits = layers.Dense(n_output_tokens, name="to_logits")(x_ln[..., -1, :])
    return tf.keras.Model(inputs=[inputs], outputs=[logits])


def embedding_summing_mlp(
    seq_len: int,
    n_input_tokens: int,
    n_output_tokens: int,
    embedding_dim: int,
    hidden_layers: Sequence[int],
    ln_pre_mlp: bool = False,
    ln_post_mlp: bool = False,
) -> tf.keras.Model:
    """Model sums embeddings and applies an MLP, in the vein of
    Liu et al. (Arxiv: 2205.10343)."""

    inputs = tf.keras.Input((None,))
    embeddings = layers.Embedding(n_input_tokens, embedding_dim)(inputs)
    x = tf.reduce_sum(embeddings, axis=1)

    if ln_pre_mlp:
        x = layers.LayerNormalization()(x)

    for n_units in hidden_layers:
        x = layers.Dense(n_units, activation="relu")(x)

    if ln_post_mlp:
        x = layers.LayerNormalization()(x)

    logits = layers.Dense(n_output_tokens, name="to_logits")(x)

    return tf.keras.Model(inputs=[inputs], outputs=[logits])


def transformer_builder(
    seq_len: int,
    n_input_tokens: int,
    n_output_tokens: int,
    params: Dict[str, Any],
) -> tf.keras.Model:
    return decoder_transformer_classifier(
        seq_len,
        n_input_tokens,
        n_output_tokens,
        params["layers"],
        params["width"],
        params["heads"],
        params["dropout"],
    )


def mlp_builder(
    ln_pre_mlp: bool, ln_post_mlp: bool
) -> Callable[[int, int, int, Dict[str, Any]], tf.keras.Model]:
    def inner_builder(
        seq_len: int,
        n_input_tokens: int,
        n_output_tokens: int,
        params: Dict[str, Any],
    ) -> tf.keras.Model:
        hidden_layers = [int(el) for el in params["hidden_layers"].split(",")]
        return embedding_summing_mlp(
            seq_len,
            n_input_tokens,
            n_output_tokens,
            params["width"],
            hidden_layers,
            ln_pre_mlp,
            ln_post_mlp,
        )

    return inner_builder


_registered_builders = {
    "transformer": transformer_builder,
    "mlp": mlp_builder(False, False),
    "mlp_pre_ln": mlp_builder(True, False),
    "mlp_post_ln": mlp_builder(False, True),
    "mlp_both_ln": mlp_builder(True, True),
}


def _prepare_embedding_weights(
    layer: tf.keras.layers.Layer,
    option: str,
) -> None:
    if not isinstance(layer, tf.keras.layers.Embedding):
        raise ValueError(f"Layer isn't an embedding layer: {type(layer)}")
    if option is None or option == "learned":
        return
    elif option == "random":
        layer.trainable = False
    elif option == "circular":
        assert len(layer.weights) == 1
        weights = layer.get_weights()[0]
        n_tokens = len(weights)
        ang_freq = 2.0 * np.pi / n_tokens
        weights[:, 0] = np.cos(ang_freq * np.arange(n_tokens))
        weights[:, 1] = np.sin(ang_freq * np.arange(n_tokens))
        layer.set_weights([weights])
        layer.trainable = False
    else:
        raise ValueError(f"Unrecognised option: {option}")


def build(
    seq_len: int,
    n_input_tokens: int,
    n_output_tokens: int,
    params: Dict[str, Any],
) -> tf.keras.Model:
    model = _registered_builders[params["model_name"]](
        seq_len, n_input_tokens, n_output_tokens, params
    )
    _prepare_embedding_weights(
        model.layers[1], params.get("embedding_weights", "learned")
    )
    return model
