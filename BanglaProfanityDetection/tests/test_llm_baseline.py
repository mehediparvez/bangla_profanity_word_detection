import src.llm_baseline as llm_mod
from src.llm_baseline import _parse_groq_response, run_llm_baseline


def test_run_llm_baseline_resumes_from_cache(tmp_path, monkeypatch):
    cache_path = str(tmp_path / "llm" / "cache.json")
    texts = ["a", "b", "c", "d"]
    sent = []
    fail_second = {"on": True}

    def fake_classify(batch, client, model, reasoning_effort):
        sent.extend(batch)
        if fail_second["on"] and batch == ["c", "d"]:
            return None  # e.g. the daily token limit
        return [1 if t in ("a", "c") else 0 for t in batch]

    monkeypatch.setattr(llm_mod, "classify_batch_groq", fake_classify)
    kwargs = dict(model_name="m", batch_size=2, delay=0, max_retries=1, cache_path=cache_path)

    first = run_llm_baseline(texts, [1, 0, 1, 0], None, **kwargs)
    assert first["successful"] == 2 and first["failed_indices"] == [2, 3]

    # The re-run only sends the batch that failed
    sent.clear()
    fail_second["on"] = False
    second = run_llm_baseline(texts, [1, 0, 1, 0], None, **kwargs)
    assert sent == ["c", "d"]
    assert second["predictions"] == [1, 0, 1, 0]
    assert second["newly_classified"] == 2

    # A cache made by another model is ignored
    sent.clear()
    run_llm_baseline(texts, [1, 0, 1, 0], None, **{**kwargs, "model_name": "other"})
    assert sent == texts


def test_parse_groq_response_valid():
    response = "1. toxic\n2. non-toxic\n3. toxic"
    labels = _parse_groq_response(response, 3)
    assert labels == [1, 0, 1]


def test_parse_groq_response_wrong_count():
    response = "1. toxic\n2. non-toxic"
    labels = _parse_groq_response(response, 3)
    assert labels is None


def test_parse_groq_response_various_formats():
    response = "1) toxic\n2) non-toxic\n3) toxic"
    labels = _parse_groq_response(response, 3)
    assert labels == [1, 0, 1]

    response = "1: toxic\n2: non-toxic\n3: non toxic"
    labels = _parse_groq_response(response, 3)
    assert labels == [1, 0, 0]


def test_parse_groq_response_reasoning_and_markdown():
    response = "<think>Comment 2 is an insult.</think>\n1. **non-toxic**\n2. Toxic\n3. Non_toxic"
    labels = _parse_groq_response(response, 3)
    assert labels == [0, 1, 0]


def test_parse_groq_response_unclosed_think_fails():
    # Reply cut off mid-reasoning: no labels, so the batch is retried rather than mis-scored
    assert _parse_groq_response("<think>Let me look at each comment", 2) is None
