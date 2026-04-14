from src.train_model import build_model


def test_build_model_shapes():
    model = build_model((60, 14), output_size=1)
    # model.input_shape is (None, 60, 14)
    assert model.input_shape[1:] == (60, 14)
    # output shape should have final dim == output_size
    assert model.output_shape[-1] == 1
