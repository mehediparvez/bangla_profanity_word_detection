from src.model import LABEL_MAP, get_tokenizer, load_model


def test_label_map():
    assert LABEL_MAP[0] == "non-toxic"
    assert LABEL_MAP[1] == "toxic"
    assert len(LABEL_MAP) == 2


def test_get_tokenizer():
    tokenizer = get_tokenizer("bert-base-multilingual-cased")
    assert tokenizer is not None
    encoded = tokenizer("test", return_tensors="pt")
    assert "input_ids" in encoded


def test_load_model():
    model = load_model("bert-base-multilingual-cased", num_labels=2)
    assert model is not None
    assert model.config.num_labels == 2
