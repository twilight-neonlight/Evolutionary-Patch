"""vLLM(OpenAI 호환) 클라이언트.

- temperature, top_p, max_tokens, seed는 키워드 필수 인자다. 기본값이 없으므로 빠뜨리면 호출이 안 된다(1.3절).
- top_k는 끄고(-1) 반복 페널티는 1.0으로 명시해 보낸다(1.4절 "top_k 없음, 반복 페널티 없음").
- 호출마다 5절 형식으로 1줄씩 로그를 남긴다. 실패한 호출도 error 필드와 함께 남긴다.
- 연결 오류·5xx는 conn_retries번까지 다시 시도한다. 이것은 operators의 유효성 재요청과 별개다.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass

import httpx

from .errors import InfraError
from .logs import JsonlLog

STAGES = ("prep", "init", "crossover", "retry", "mutation")
# retry는 그것을 일으킨 원래 단계의 샘플링 값을 쓴다()
RETRY_ORIGINS = ("init", "crossover", "mutation")
TOP_K_DISABLED = -1
REPETITION_PENALTY_NONE = 1.0


class LLMConnectionError(InfraError):
    """재시도를 모두 써도 서버에 닿지 못함 (인프라 오류, B5)."""


class LLMRequestError(RuntimeError):
    """서버가 요청을 거절함(4xx, 컨텍스트 초과 제외)."""


class ContextBudgetExceeded(RuntimeError):
    """입력 + max_tokens가 컨텍스트 창을 넘음(4.7: 감지만 하고 자르지 않음)."""

    def __init__(self, prompt_tokens: int | None, max_tokens: int, context_window: int):
        super().__init__(
            f"budget exceeded: prompt={prompt_tokens} + max_tokens={max_tokens} > {context_window}"
        )
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.context_window = context_window


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    latency_s: float
    log_row: dict


def _require_number(name: str, value, *, integer: bool = False) -> None:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name}는 명시해야 합니다")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name}는 정수여야 합니다: {value!r}")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}가 숫자가 아닙니다: {value!r}")


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        context_window: int,
        http_timeout_s: float,
        conn_retries: int,
        conn_backoff_s: float,
        call_log: JsonlLog,
        run_id: str,
        response_log: JsonlLog | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        # /tokenize는 /v1 밖에 있다
        self.server_root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        self.model = model
        self.context_window = context_window
        self.conn_retries = conn_retries
        self.conn_backoff_s = conn_backoff_s
        self.call_log = call_log
        self.response_log = response_log  # 응답 원문 (call_id로 연결, 파일럿 점검용)
        self.run_id = run_id
        self._sleep = sleep
        self._http = httpx.Client(timeout=http_timeout_s, transport=transport)

    @classmethod
    def from_config(cls, cfg, *, call_log: JsonlLog, run_id: str, **kw) -> "LLMClient":
        return cls(
            base_url=cfg.require("llm.base_url"),
            model=cfg.require("llm.model"),
            context_window=cfg.require("llm.context_window"),
            http_timeout_s=cfg.require("llm.http_timeout_s"),
            conn_retries=cfg.require("llm.conn_retries"),
            conn_backoff_s=cfg.require("llm.conn_backoff_s"),
            call_log=call_log,
            run_id=run_id,
            **kw,
        )

    # ---- 내부: 연결 재시도 ----
    def _post(self, url: str, payload: dict) -> tuple[httpx.Response, int]:
        attempts = 0
        last_err: Exception | None = None
        while attempts <= self.conn_retries:
            attempts += 1
            try:
                resp = self._http.post(url, json=payload)
            except httpx.TransportError as e:
                last_err = e
            else:
                if resp.status_code < 500:
                    return resp, attempts
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if attempts <= self.conn_retries:
                self._sleep(self.conn_backoff_s * attempts)
        raise LLMConnectionError(f"{url}: {attempts}회 시도 실패: {last_err}")

    def count_tokens(self, messages: list[dict]) -> int:
        """서버의 /tokenize로 채팅 템플릿을 적용한 입력 토큰 수를 센다(1.3절)."""
        payload = {"model": self.model, "messages": messages, "add_generation_prompt": True}
        resp, _ = self._post(f"{self.server_root}/tokenize", payload)
        if resp.status_code != 200:
            raise LLMRequestError(f"/tokenize HTTP {resp.status_code}: {resp.text[:300]}")
        return int(resp.json()["count"])

    def chat(
        self,
        messages: list[dict],
        *,
        stage: str,
        instance_id: str,
        generation: int | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        retry_index: int = 0,
        retry_of: str | None = None,
        extra_log: dict | None = None,
    ) -> LLMResult:
        if stage not in STAGES:
            raise ValueError(f"알 수 없는 stage: {stage}")
        if stage == "retry" and retry_of not in RETRY_ORIGINS:
            raise ValueError(f"stage=retry에는 retry_of{RETRY_ORIGINS}가 필요합니다: {retry_of!r}")
        if stage != "retry" and retry_of is not None:
            raise ValueError("retry_of는 stage=retry에서만 씁니다")
        _require_number("temperature", temperature)
        _require_number("top_p", top_p)
        _require_number("max_tokens", max_tokens, integer=True)
        _require_number("seed", seed, integer=True)

        row = {
            "run_id": self.run_id,
            "instance_id": instance_id,
            "stage": stage,
            "generation": generation,
            "model": self.model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "latency_s": None,
            "retry_index": retry_index,
            # 5절 밖의 보조 필드
            "retry_of": retry_of,
            # A: 적용 상한 기록. init 계열은 operators가 extra_log로 덮어쓴다(공식값·컨텍스트값·감소 여부)
            "max_tokens_capped_by_context": False,
            "top_k": TOP_K_DISABLED,
            "repetition_penalty": REPETITION_PENALTY_NONE,
            "conn_attempts": 0,
            "call_id": uuid.uuid4().hex[:16],
            "error": None,
            "ts": time.time(),
            **(extra_log or {}),
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "top_k": TOP_K_DISABLED,
            "repetition_penalty": REPETITION_PENALTY_NONE,
        }

        t0 = time.monotonic()
        try:
            # 컨텍스트 초과는 보내기 전에 감지한다(4.7)
            n_in = self.count_tokens(messages)
            if n_in + max_tokens > self.context_window:
                raise ContextBudgetExceeded(n_in, max_tokens, self.context_window)
            resp, attempts = self._post(f"{self.base_url}/chat/completions", payload)
            row["conn_attempts"] = attempts
            if resp.status_code != 200:
                body = resp.text[:500]
                if resp.status_code == 400 and ("max_model_len" in body or "context length" in body):
                    raise ContextBudgetExceeded(n_in, max_tokens, self.context_window)
                raise LLMRequestError(f"HTTP {resp.status_code}: {body}")
            data = resp.json()
        except ContextBudgetExceeded as e:
            row.update(error="budget_exceeded", prompt_tokens=e.prompt_tokens, completion_tokens=0,
                       latency_s=round(time.monotonic() - t0, 3))
            self.call_log.append(row)
            raise
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {e}"[:500], latency_s=round(time.monotonic() - t0, 3))
            self.call_log.append(row)
            raise

        latency = time.monotonic() - t0
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        row.update(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            latency_s=round(latency, 3),
        )
        self.call_log.append(row)
        text = choice["message"].get("content") or ""
        if self.response_log is not None:
            self.response_log.append({"call_id": row["call_id"], "instance_id": instance_id, "stage": stage,
                                      "messages": messages, "response": text})
        return LLMResult(
            text=text,
            prompt_tokens=row["prompt_tokens"] or 0,
            completion_tokens=row["completion_tokens"] or 0,
            finish_reason=row["finish_reason"],
            latency_s=latency,
            log_row=row,
        )

    def close(self) -> None:
        self._http.close()
