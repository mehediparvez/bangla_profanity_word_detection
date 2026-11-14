from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_MAP = {0: "non-toxic", 1: "toxic"}


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name, num_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model


def load_trained_model(model_path, num_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
