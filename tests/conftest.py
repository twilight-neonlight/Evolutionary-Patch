import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evopatch.config import Config  # noqa: E402
from evopatch.llm_client import LLMResult  # noqa: E402
from evopatch.logs import JsonlLog  # noqa: E402
from evopatch.types import Location  # noqa: E402

ORIG = '''import os


def add(a, b):
    """Add."""
    total = a + b
    return total


def other():
    return 1


class K:
    def m(self, x):
        if x:
            return x
        return None
'''


@pytest.fixture
def cfg():
    return Config.load(ROOT / "tests" / "data" / "test_config.yaml")


@pytest.fixture
def originals():
    return {"pkg/mod.py": ORIG}


@pytest.fixture
def locations():
    lines = ORIG.splitlines()
    def loc(sym, s, e):
        return Location(file="pkg/mod.py", symbol=sym, start_line=s, end_line=e, code="\n".join(lines[s - 1:e]))
    return [loc("add", 4, 7), loc("K.m", 15, 18)]


class FakeLLM:
    """모의 LLM: 미리 정한 응답을 차례로 돌려주고, 받은 요청을 기록한다."""

    def __init__(self, responses, prompt_tokens=100):
        self.responses = list(responses)
        self.calls = []
        self.run_id = "test"
        self.prompt_tokens = prompt_tokens

    def count_tokens(self, messages):
        return self.prompt_tokens

    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        text = self.responses.pop(0)
        return LLMResult(text=text, prompt_tokens=10, completion_tokens=5, finish_reason="stop",
                         latency_s=0.0, log_row={})


@pytest.fixture
def make_ctx(cfg, originals, locations, tmp_path):
    from evopatch.operators import OpContext

    def _mk(responses, prompt_tokens=100):
        llm = FakeLLM(responses, prompt_tokens)
        ctx = OpContext(cfg=cfg, llm=llm, instance_id="x__y-1", run_index=0, problem_statement="bug",
                        locations=locations, originals=originals, validity_log=JsonlLog(tmp_path / "v.jsonl"))
        return ctx, llm
    return _mk
