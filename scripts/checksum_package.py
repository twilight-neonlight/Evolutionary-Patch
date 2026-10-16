"""제3자 연구 재현 테스트 패키지의 모든 파일 SHA-256 목록을 만든다(목록은 git 제외 위치에 둔다).
출력 목록 형식은 `sha256sum`과 같다(SHA-256 두 칸 공백 패키지 기준 상대 경로, 경로순). 목록 파일의 SHA-256을 출력하며,
그 값을 설정 external.repro_tests_checksum에, 목록 경로를 external.repro_tests_checksum_list에 넣는다.
실행: python scripts/checksum_package.py --dir external/<패키지> --out external/<패키지>.sha256
"""
import argparse
import hashlib
from pathlib import Path


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.dir).resolve()
    out = Path(a.out).resolve()
    if out.is_relative_to(root):
        raise SystemExit("목록 파일은 패키지 폴더 밖에 둔다")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    out.write_text("".join(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n" for p in files))
    print(f"files={len(files)} list_sha256={sha256(out)}")


if __name__ == "__main__":
    main()
