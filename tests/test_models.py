import tensorflow as tf  # type: ignore
from grokking.models import transformer_layer, decoder_transformer_classifier


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
