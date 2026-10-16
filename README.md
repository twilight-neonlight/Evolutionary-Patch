# 테스트 적합도를 쓰는 LLM 기반 진화적 패치 생성

## 개요

이슈마다 재현 테스트와 회귀 테스트의 실행 결과를 적합도로 삼아, LLM이 만든 패치 후보 개체군을 진화시킨다.
전처리에서 재현 테스트(제3자 연구 산출물, 쓸 수 없으면 LLM 생성), 회귀 테스트(Agentless 사전 계산 목록), 위치(파일·함수 최대 3개)를 정한다.
한 번의 호출로 서로 다른 접근의 패치 N=10개를 만들고(0세대), 각 후보를 Docker에서 실행해 적합도 (r, g)
(r = 재현 테스트 통과, g = 통과한 회귀 테스트 수)로 순위를 매긴다. 세대마다 순위 비례로 부모 쌍 N/2개를 뽑아
교배(재시도 최대 2회)와 변이(확률 0.15)로 자식 N/2개를 만들고, 상위 N/2와 합쳐 다음 세대를 이룬다.
완전 통과 후보가 나오면 조기 종료하고, K=5세대(0세대 포함)에서 최상위를 제출한다. 방법의 기준 명세는 `docs/METHOD.md`.

## 논문 절과 코드 위치

| 방법 | 코드 |
|---|---|
| 전처리(재현 테스트·회귀 테스트·위치, 대체 순서) | `src/evopatch/prep/hybrid_v2.py`(+ `prep/hybrid.py`, `prep/simple.py`, `prep/lab_fixed.py`의 도우미), `specs.py`(회귀 목록), `locate.py` |
| 초기 개체군(0세대, 부족분 재요청) | `src/evopatch/operators.py` `initial_population` |
| 적합도 평가(실행, r·g, 같은 패치 재사용) | `src/evopatch/executor.py`, `runner.py` `_evolve.evaluate`, `population.py` `Individual.fitness` |
| 종료 확인(조기 종료·K) | `src/evopatch/loop.py` `evolve` |
| 부모 선택(순위 비례, 동점 평균) | `src/evopatch/population.py` `tied_rank_probabilities`, `draw_parent_pairs` |
| 교배·재시도 | `src/evopatch/operators.py` `crossover` |
| 변이 | `src/evopatch/operators.py` `mutate` |
| 세대 교체(상위 N/2 + 자식 N/2) | `src/evopatch/loop.py`, `population.py` `select_elites` |
| 유효성 검사(위치 범위·컴파일·빈 diff) | `src/evopatch/edits.py` |
| 부모 실행 결과(교배·변이 입력) | `src/evopatch/feedback.py`, `regmsg.py`(회귀 오류 내용) |
| 프롬프트 | `prompts/prompts.txt`(문장), `src/evopatch/prompts.py`·`promptdoc.py`(조립) |
| 채점 | `src/evopatch/grading/` |

## 환경 준비

Python 3.12.3, Docker, (모델 서버용) GPU가 필요하다.

```bash
python3.12 -m venv .venv
.venv/bin/pip install --require-hashes --no-deps -r requirements.lock   # swebench 5.0.2 포함, 해시 고정
cp config/main.yaml.example config/main.yaml
cp config/local.yaml.example config/local.yaml                         # 서버별 값(git 제외)
export PYTHONPATH=src
```

## 데이터 받기

```bash
.venv/bin/python scripts/prepare_data.py --hf-cache
```

SWE-bench Lite·Verified(Hugging Face 고정 revision), Agentless 회귀 목록(GitHub 릴리스 v1.5.0), 실행 명세(받은 데이터셋에서 생성)를
`data/`에 만들고 SHA-256을 검증한다. 출처·버전·해시는 `data/MANIFEST`. 파이프라인 로더는 허용 6필드(instance_id, repo,
base_commit, problem_statement, version, environment_setup_commit)만 읽고, 정답 필드는 채점·분석 코드만 읽는다.

**제3자 연구 재현 테스트 산출물은 이 저장소에 없다(커밋하지 않는다).** 따로 받아 git 제외 위치(예: `external/`)에 두고
`config/local.yaml`에 적는다.

```yaml
external:
  repro_tests_format: new_standalone        # <이슈>/{generated_test.py, code_context.json, issue_clue.json}
  repro_tests_dir: external/<패키지>
  repro_tests_checksum_list: external/<패키지>.sha256
  repro_tests_checksum: <목록 파일의 SHA-256>
```

체크섬 목록은 `python scripts/checksum_package.py --dir external/<패키지> --out external/<패키지>.sha256`로 만든다(출력의
`list_sha256`을 `repro_tests_checksum`에). 실행 시작 때 목록 해시와 패키지 파일 전체를 대조하고 실행 메타에 기록한다.
`new_standalone`은 `generated_test.py`를 LLM 재현 테스트와 같은 저장소별 고정 경로(Django `tests/swe_repro/tests.py`,
Sympy `sympy/core/tests/test_swe_repro.py`, 그 밖 `prep.simple.repro_paths`)에 두고 파일 전체를 실행한다. 제3자 테스트가 1순위이며,
파싱 불가·base 실행 오류·시간 초과·import 오류·수집 0이면 LLM 테스트로 대체하고 사유를 이슈별로 기록한다.
기존 형식(기존 테스트 파일 + 덧붙인 블록)은 `repro_tests_format: old_appended`. 채택·대체 수 점검:
`python scripts/check_repro_tests.py --out reports/repro_tests_check`. 산출물이 없으면 모든 이슈가 LLM 재현 테스트로 진행된다.

## 모델 서빙

```bash
vllm serve <모델 경로> --served-model-name devstral --max-model-len 32768 --port 8000
```

설정 `llm.model`은 `--served-model-name`과 같아야 한다(시작 전에 `/v1/models`로 확인). 토큰 수는 서버 `/tokenize`로 센다.

## 실행

```bash
.venv/bin/python -m evopatch.cli --config config/main.yaml --ids-file <이슈 ID 목록> --run-id <이름>
# 이어 실행: --run-dir runs/<날짜>_<이름> --resume
```

결과는 `runs/<날짜>_<이름>/`(issues.jsonl, candidates.jsonl, llm_calls.jsonl, responses.jsonl, predictions.jsonl, run_meta.json 등).

## 채점

```bash
.venv/bin/python -m evopatch.grading.aggregate --run-dir runs/<날짜>_<이름> --by-set --per-issue
# 끝나지 않은 실행의 완료 이슈만: python -m evopatch.grading.partial --run-dir ... --per-issue
```

swebench 5.0.2 하니스로 채점한다. 해결률 분모는 데이터셋 전체(Lite 300, Verified 500)이며, 이슈별 해결 여부는 실행 폴더의
`resolved_per_issue.json`에 저장된다.

## K=1/3 어블레이션

K=5 실행 기록을 해당 세대에서 끊은 제출 패치를 채점한다(새 LLM 호출 없음).

```bash
.venv/bin/python scripts/ablation_k.py --run-dir runs/<K=5 실행> --k 1 --k 3
.venv/bin/python -m evopatch.grading.aggregate --run-dir runs/<K=5 실행>__K1 --by-set --per-issue
```

## 재현성 정보

- 프롬프트: `prompts/prompts.txt`(읽는 법·호출별 조립 순서는 `prompts/README.md`). 실행 메타에 문서 경로·SHA-256, 호출 로그에
  호출마다 렌더링된 프롬프트 SHA-256과 쓴 구간 이름이 남는다.
- 시드: 모든 무작위성은 `derive_seed(instance_id, run_index, …)`(SHA-256 앞 4바이트)로 유도한다. LLM 호출 seed, 부모 선택,
  변이 판정이 이슈·세대·자식 번호별로 정해진다.
- 설정: 결과에 영향을 주는 값은 `config/main.yaml.example`에 있고, 실행 폴더에 합친 설정(`config.snapshot.yaml`)과
  코드 스냅샷·코드/데이터/패키지 해시(`run_meta.json`)가 남는다. 의존성은 `requirements.lock`(해시 고정).
- 미결정 항목(설정으로 전환): 프롬프트 타입(`prompts.path`), 수정 범위 S0/S2(`algorithm.edit_scope`), 백본(`llm.model`).

## 테스트

```bash
.venv/bin/python -m pytest -q tests
```

`tests/equivalence/`의 재생 검증은 기준 실행 기록이 있을 때만 쓴다(`EVOPATCH_PILOT_RUN` 등 환경 변수가 없으면 쓰지 않는다).

## 알려진 한계

- 제3자 연구 재현 테스트 산출물은 별도로 받아야 하며, 없으면 재현 테스트를 LLM으로 만든다.
- LLM 응답은 vLLM의 seed를 써도 하드웨어·배치에 따라 완전히 같게 재현되지 않을 수 있다.
- 실행 환경은 SWE-bench 공식 Docker 이미지에 의존한다(이미지를 받을 수 없으면 그 이슈는 빈 패치로 기록).
- 토큰 수 추정(피드백 길이 검사)은 문자 수 기반 근사다.

## 라이선스

라이선스 결정 예정.
