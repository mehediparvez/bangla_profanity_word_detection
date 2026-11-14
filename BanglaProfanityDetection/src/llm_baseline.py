import hashlib
import os
import re
import time
import json

from .model import LABEL_MAP


def _text_key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path, model_name):
    if not cache_path or not os.path.exists(cache_path):
        return {}
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("model") != model_name:
        # Labels from a different model must not be mixed into this run
        print(f"Ignoring cache from model {data.get('model')!r} (now using {model_name!r})")
        return {}
    return data.get("labels", {})


def _save_cache(cache_path, model_name, cache):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "labels": cache}, f)
    os.replace(tmp_path, cache_path)  # never leave a half-written cache behind


def _parse_groq_response(response_text, num_comments):
    # Reasoning models (e.g. Qwen) may prepend their thinking in <think> tags
    response_text = re.sub(r"<think>.*?(</think>|$)", "", response_text or "", flags=re.S | re.I)
    labels = []
    for line in response_text.strip().split("\n"):
        # Ignore markdown emphasis such as "1. **toxic**"
        line = line.strip().replace("*", "").replace("`", "")
        match = re.match(r"^\d+\s*[\.\):\-]\s*(non[\s_\-]?toxic|toxic)\b", line, re.IGNORECASE)
        if match:
            label_str = match.group(1).lower()
            labels.append(1 if label_str == "toxic" else 0)
    if len(labels) != num_comments:
        print(f"Could not parse {num_comments} labels (got {len(labels)}). "
              f"Reply started with: {response_text.strip()[:200]!r}")
        return None
    return labels


def classify_batch_groq(texts, client, model="llama-3.3-70b-versatile", reasoning_effort=None):
    prompt_lines = [f"{i+1}. {t}" for i, t in enumerate(texts)]
    prompt = (
        "Classify each Bangla comment as 'toxic' or 'non-toxic'. "
        "Reply with ONLY a numbered list like:\n"
        "1. toxic\n2. non-toxic\n...\n\n"
        "Comments:\n" + "\n".join(prompt_lines)
    )

    extra = {}
    if reasoning_effort:
        # e.g. "none" turns off thinking on Qwen, leaving the token budget for the answer
        extra["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        # A 25-line label list needs ~200 tokens; the free tier caps output at 1,000 tokens/minute
        max_tokens=512,
        **extra,
    )
    return _parse_groq_response(response.choices[0].message.content, len(texts))


def classify_batch_gemini(texts, model):
    prompt_lines = [f"{i+1}. {t}" for i, t in enumerate(texts)]
    prompt = (
        "Classify each Bangla comment as 'toxic' or 'non-toxic'. "
        "Reply with ONLY a numbered list like:\n"
        "1. toxic\n2. non-toxic\n...\n\n"
        "Comments:\n" + "\n".join(prompt_lines)
    )

    response = model.generate_content(prompt)
    return _parse_groq_response(response.text, len(texts))


def run_llm_baseline(
    texts,
    labels,
    client,
    model_name="llama-3.3-70b-versatile",
    batch_size=25,
    delay=2,
    max_retries=3,
    provider="groq",
    reasoning_effort=None,
    cache_path=None,
):
    """Classify texts in batches.

    With cache_path, every finished batch is saved straight away and reused on the
    next call, so a re-run (after a disconnect or the daily token limit) only sends
    the comments that have no label yet.
    """
    cache = _load_cache(cache_path, model_name)
    all_preds = [cache.get(_text_key(t)) for t in texts]
    if cache_path:
        done = sum(p is not None for p in all_preds)
        print(f"Cache: {done}/{len(texts)} comments already classified, "
              f"{len(texts) - done} to go")
    failed_indices = []
    newly_classified = 0
    start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_idx = list(range(i, min(i + batch_size, len(texts))))
        if all(all_preds[j] is not None for j in batch_idx):
            continue

        success = False
        for attempt in range(max_retries):
            try:
                if provider == "groq":
                    batch_labels = classify_batch_groq(
                        batch_texts, client, model_name, reasoning_effort
                    )
                else:
                    batch_labels = classify_batch_gemini(batch_texts, client)

                if batch_labels is not None:
                    for idx, label in zip(batch_idx, batch_labels):
                        all_preds[idx] = label
                        cache[_text_key(texts[idx])] = label
                    _save_cache(cache_path, model_name, cache)
                    newly_classified += len(batch_idx)
                    success = True
                    break
            except Exception as e:
                print(f"Batch {i//batch_size + 1} attempt {attempt+1} failed: {e}")
                time.sleep(delay * (attempt + 1))

        if not success:
            failed_indices.extend(batch_idx)
            print(f"Batch {i//batch_size + 1} failed after {max_retries} retries")

        if i + batch_size < len(texts):
            time.sleep(delay)

    valid_idx = [i for i in range(len(texts)) if all_preds[i] is not None]
    valid_preds = [all_preds[i] for i in valid_idx]
    valid_labels = [labels[i] for i in valid_idx]

    # Cached comments cost no time, so the speed is measured on this call's comments only and
    # scaled to all of them; otherwise a resumed run would report an inflated speed
    elapsed = time.time() - start_time
    inference_time = (
        elapsed / newly_classified * len(valid_preds) if newly_classified else None
    )

    return {
        "inference_time": inference_time,
        "newly_classified": newly_classified,
        "predictions": all_preds,
        "valid_predictions": valid_preds,
        "valid_labels": valid_labels,
        "failed_indices": failed_indices,
        "total": len(texts),
        "successful": len(valid_preds),
    }


def run_better_profanity_baseline(texts, labels):
    from better_profanity import profanity

    profanity.load_censor_words()
    preds = []
    for text in texts:
        preds.append(1 if profanity.contains_profanity(text) else 0)

    return preds
