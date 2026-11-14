import torch
from .model import LABEL_MAP


def predict_text(text, model, tokenizer, max_length=256):
    model.eval()
    device = next(model.parameters()).device

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()

    return {
        "text": text,
        "label": LABEL_MAP[pred],
        "label_id": pred,
        "confidence": confidence,
        "probabilities": {
            LABEL_MAP[i]: probs[0][i].item() for i in range(len(LABEL_MAP))
        },
    }


def predict_batch(texts, model, tokenizer, max_length=256, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            results.append(predict_text(text, model, tokenizer, max_length))
    return results
