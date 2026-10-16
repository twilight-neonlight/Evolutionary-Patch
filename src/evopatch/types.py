""" 인터페이스 계약. 정답 필드는 여기 없다."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Location:
    file: str
    symbol: str
    start_line: int  # 1-based, 포함
    end_line: int  # 1-based, 포함
    code: str  # base commit 기준 함수 본문


@dataclass(frozen=True)
class ReproTest:
    path: str  # 저장소 기준 경로 (Django는 tests.py 경로)
    content: str
    run_target: str  # 테스트 명령 앞부분 뒤에 붙는 대상(경로 또는 Django 라벨)


@dataclass(frozen=True)
class LLMUsage:
    model: str
    stage: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class PrepOutput:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    locations: list[Location]
    repro_test: ReproTest
    regression_tests: list[str]  # 이슈별 첫 줄의 tests_passing_in_original_repo
    test_files: list[str]  # test_patch에서 오프라인 추출. 실행기만 읽는다
    prep_usage: list[LLMUsage]
    prep_provider: str  # 4.1a: 어떤 전처리로 얻은 결과인지

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "PrepOutput":
        return cls(
            **{
                **d,
                "locations": [Location(**l) for l in d["locations"]],
                "repro_test": ReproTest(**d["repro_test"]),
                "prep_usage": [LLMUsage(**u) for u in d["prep_usage"]],
            }
        )


class PrepFailure(RuntimeError):
    """전처리 실패(4.1a: 미해결로 집계). 그때까지 쓴 LLM 사용량을 함께 담는다."""

    def __init__(self, reason: str, usage: list[LLMUsage], detail: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.usage = usage
        self.detail = detail or {}


@dataclass(frozen=True)
class Edit:
    file: str
    search: str
    replace: str


@dataclass(frozen=True)
class Candidate:
    edits: tuple[Edit, ...]
    patch_hash: str  # 정규화한 패치 텍스트 해시
    diff: str = ""  # 원본 파일 전체와 비교한 unified diff (4.3)


@dataclass
class ExecResult:
    repro_pass: bool
    reg_passed: int
    reg_total: int
    timed_out: bool
    output_path: str
    # 보조 정보
    repro_statuses: dict[str, str] = field(default_factory=dict)  # 하네스 파서 상태(재현 파일 테스트만)
    repro_timed_out: bool = False
    reg_timed_out: bool = False
    apply_ok: bool = True
    no_regression_signal: bool = False
    error: str | None = None
    duration_s: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)
