import pytest
import tensorflow as tf  # type: ignore
from grokking.models import (
    transformer_layer,
    decoder_transformer_classifier,
    embedding_summing_mlp,
)


class TestTransformerLayer:
    def test_input_output_shapes(self) -> None:
        seq_len = 5
        attention_mask = tf.ones((5, 5))
        width = 32
        heads = 4
        batch_size = 3

        inputs = tf.random.normal((batch_size, seq_len, width))
        outputs = transformer_layer(inputs, width, heads, attention_mask)
        assert list(outputs.shape) == [batch_size, seq_len, width]


class TestDecoderTransformerClassifier:
    def test_input_output_shapes(self) -> None:
        n_classes = 4
        n_layers = 2
        width = 32
        heads = 4

        inputs = tf.constant(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        )
        batch_size, seq_len = inputs.shape
        vocabulary_size = 7  # max token is 6

        model = decoder_transformer_classifier(
            seq_len, vocabulary_size, n_classes, n_layers, width, heads
        )
        logits = model(inputs)

        assert list(logits.shape) == [batch_size, n_classes]


class TestEmbeddingSummingMLP:
    def test_input_output_shapes(self) -> None:
        n_input_tokens = 4
        n_output_tokens = 6
        embedding_dim = 2
        hidden_layers = [32, 32]

        inputs = tf.constant([[0, 1], [2, 3], [0, 3]])
        batch_size, seq_len = inputs.shape

        model = embedding_summing_mlp(
            seq_len,
            n_input_tokens,
            n_output_tokens,
            embedding_dim,
            hidden_layers,
        )
        model.summary()
        assert len(model.layers) == 6  # input, embed, sum, hidden, logits
        logits = model(inputs)

        assert list(logits.shape) == [batch_size, n_output_tokens]

    @pytest.mark.parametrize(
        "pre,post,n_layers",
        [
            (False, False, 6),
            (False, True, 7),
            (True, False, 7),
            (True, True, 8),
        ],
    )
    def test_ln_options(self, pre: bool, post: bool, n_layers: int) -> None:
        model = embedding_summing_mlp(
            2, 4, 4, 1, [8, 8], ln_pre_mlp=pre, ln_post_mlp=post
        )
        assert len(model.layers) == n_layers
