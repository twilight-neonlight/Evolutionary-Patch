"""무인 실행용 오류 구분 (B4·B5·B6).

InfraError: 풀이와 무관한 인프라 문제(vLLM 연결 불가, 도커 데몬 오류, 디스크 부족). 이슈 실패로 세지 않고 재시도·중단.
IssueTimeout: 이슈 하나의 wall-clock 상한 초과. issue_timeout으로 기록하고 다음 이슈로.
ImageUnavailable: 이미지를 받을 수 없는 이슈(Docker Hub 404 등). image_unavailable로 기록하고 다음 이슈로 (A2).
"""


class InfraError(RuntimeError):
    pass


class IssueTimeout(RuntimeError):
    pass


class ImageUnavailable(RuntimeError):
    """A2: 이슈의 실행 이미지를 받을 수 없음(Docker Hub에 없음/404). 그 이슈만 image_unavailable로 기록하고 다음 이슈로.
    cause는 원인 분류(manifest_unknown / repository_not_found / not_found). 서로 다른 이슈에서 같은 원인이
    연속 image_unavailable_escalate_after건이면 cli가 infra_error로 승격한다."""

    def __init__(self, cause: str, message: str):
        super().__init__(f"{cause}: {message}")
        self.cause = cause


class DiskGuardStop(InfraError):
    """A1: 디스크 여유가 unattended.disk_guard_gb 미만. 대기하지 않고 깨끗이 중단(종료 코드 4)."""
