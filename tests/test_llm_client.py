"""8.1-1 검증: 요청마다 샘플링 파라미터 명시·로그, 연결 재시도, 컨텍스트 초과 감지."""
import json

import httpx
import pytest

from evopatch.llm_client import (ContextBudgetExceeded, LLMClient, LLMConnectionError)
from evopatch.logs import JsonlLog

LOG_FIELDS = ["run_id", "instance_id", "stage", "generation", "model", "temperature", "top_p",
              "max_tokens", "seed", "prompt_tokens", "completion_tokens", "finish_reason",
              "latency_s", "retry_index"]


class FakeServer:
    def __init__(self, fail_first=0, prompt_count=100, status=200):
        self.fail_first = fail_first
        self.prompt_count = prompt_count
        self.status = status
        self.chat_payloads = []
        self.calls = 0

    def __call__(self, request: httpx.Request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": self.prompt_count})
        self.calls += 1
        if self.calls <= self.fail_first:
            raise httpx.ConnectError("refused", request=request)
        self.chat_payloads.append(body)
        if self.status != 200:
            return httpx.Response(self.status, text="boom")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": self.prompt_count, "completion_tokens": 7}})


def make(tmp_path, server, retries=3):
    log = JsonlLog(tmp_path / "llm_calls.jsonl")
    c = LLMClient(base_url="http://x/v1", model="m", context_window=1000, http_timeout_s=5,
                  conn_retries=retries, conn_backoff_s=0, call_log=log, run_id="r1",
                  transport=httpx.MockTransport(server), sleep=lambda s: None)
    return c, log


KW = dict(stage="init", instance_id="i1", generation=0, temperature=0.7, top_p=1.0, max_tokens=50, seed=11)


def test_params_sent_and_logged(tmp_path):
    srv = FakeServer()
    c, log = make(tmp_path, srv)
    r = c.chat([{"role": "user", "content": "hi"}], **KW)
    assert r.text == "ok"
    p = srv.chat_payloads[0]
    assert (p["temperature"], p["top_p"], p["max_tokens"], p["seed"]) == (0.7, 1.0, 50, 11)
    assert p["top_k"] == -1 and p["repetition_penalty"] == 1.0
    row = log.read()[0]
    for f in LOG_FIELDS:
        assert f in row
    assert row["temperature"] == 0.7 and row["prompt_tokens"] == 100 and row["completion_tokens"] == 7
    assert row["finish_reason"] == "stop" and row["error"] is None


@pytest.mark.parametrize("missing", ["temperature", "top_p", "max_tokens", "seed"])
def test_sampling_params_mandatory(tmp_path, missing):
    c, _ = make(tmp_path, FakeServer())
    kw = dict(KW)
    del kw[missing]
    with pytest.raises(TypeError):
        c.chat([], **kw)
    kw[missing] = None
    with pytest.raises(ValueError):
        c.chat([], **kw)


def test_bad_stage(tmp_path):
    c, _ = make(tmp_path, FakeServer())
    with pytest.raises(ValueError):
        c.chat([], **{**KW, "stage": "elite"})


def test_connection_retry_then_success(tmp_path):
    srv = FakeServer(fail_first=2)
    c, log = make(tmp_path, srv, retries=3)
    c.chat([], **KW)
    assert log.read()[0]["conn_attempts"] == 3


def test_connection_retry_exhausted_logged(tmp_path):
    c, log = make(tmp_path, FakeServer(fail_first=99), retries=2)
    with pytest.raises(LLMConnectionError):
        c.chat([], **KW)
    row = log.read()[0]
    assert row["error"].startswith("LLMConnectionError")


def test_budget_exceeded_detected_before_send(tmp_path):
    srv = FakeServer(prompt_count=990)
    c, log = make(tmp_path, srv)
    with pytest.raises(ContextBudgetExceeded):
        c.chat([], **KW)
    assert srv.chat_payloads == []
    assert log.read()[0]["error"] == "budget_exceeded"


def test_log_appends_not_overwrites(tmp_path):
    c, log = make(tmp_path, FakeServer())
    c.chat([], **KW)
    c2, log2 = make(tmp_path, FakeServer())
    c2.chat([], **KW)
    assert len(log2.read()) == 2


def test_retry_requires_origin(tmp_path):
    c, _ = make(tmp_path, FakeServer())
    with pytest.raises(ValueError):
        c.chat([], **{**KW, "stage": "retry"})
    with pytest.raises(ValueError):
        c.chat([], **{**KW, "retry_of": "init"})  # stage=init인데 retry_of 지정


@pytest.mark.parametrize("origin,temp", [("init", 0.7), ("crossover", 0.5), ("mutation", 0.7)])
def test_retry_temperature_follows_origin(tmp_path, origin, temp):
    from pathlib import Path
    from evopatch.config import Config, sampling_params
    cfg = Config.load(Path(__file__).resolve().parents[1] / "tests" / "data" / "test_config.yaml")
    sp = sampling_params(cfg, "retry", origin)
    assert sp == {"temperature": temp, "top_p": 1.0}
    srv = FakeServer()
    c, log = make(tmp_path, srv)
    c.chat([], **{**KW, "stage": "retry", "retry_of": origin, **sp})
    assert srv.chat_payloads[0]["temperature"] == temp
    row = log.read()[0]
    assert row["stage"] == "retry" and row["retry_of"] == origin and row["temperature"] == temp
