"""PrepProvider 인터페이스 . 다른 모듈은 make_provider로만 구현체를 얻는다."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..data_loader import InstanceView
from ..specs import ExecSpec
from ..types import PrepOutput


@dataclass
class PrepDeps:
    cfg: object
    llm: object  # LLMClient
    executor: object  # Executor (base commit 파일 읽기)
    exec_specs: dict[str, ExecSpec]
    regression_pool: dict[str, list[str]]
    run_index: int
    extra_log: dict = field(default_factory=dict)
    work_dir: object = None  # Path|None: 전처리 중 실행기 출력 폴더(hybrid의 제3자 연구 테스트 base 실행). 다른 구현체는 안 씀


class PrepProvider(ABC):
    name: str = ""

    def __init__(self, deps: PrepDeps):
        self.deps = deps
        self.meta: dict = {}  # 로그용 부가 정보 (요청/찾은 위치 수 등)

    @abstractmethod
    def prepare(self, view: InstanceView) -> PrepOutput: ...

    # 공통: 회귀 목록(1.5)과 test_files(4.6a, 오프라인 산출물)
    def regression_tests(self, iid: str) -> list[str]:
        return list(self.deps.regression_pool.get(iid, []))

    def test_files(self, iid: str) -> list[str]:
        return list(self.deps.exec_specs[iid].test_files)


def make_provider(name: str, deps: PrepDeps) -> PrepProvider:
    if name == "simple":
        from .simple import SimplePrep
        return SimplePrep(deps)
    if name == "hybrid_v2":  # 외부 재현 테스트 우선 → base 실행 → V2 위치 호출 → 대체(METHOD 2절)
        from .hybrid_v2 import HybridV2Prep
        return HybridV2Prep(deps)
    raise ValueError(f"알 수 없는 prep.provider: {name} (simple | hybrid_v2)")
