# 프롬프트 고정 문서

모든 LLM 호출의 프롬프트 문장은 `prompts/prompts.txt` 하나에만 있다. 코드(`src/evopatch/prompts.py`, `prep/simple.py`,
`prep/hybrid_v2.py`, `feedback.py`, `regmsg.py`, `scope.py`, `redact.py`)는 구간을 이름으로 읽고, 상황에 맞는 구간을 고르고,
`{{자리표시자}}`를 채우고, 아래 순서로 이은 뒤 기존 규칙대로 test_files 마스킹과 길이 검사를 할 뿐이다.
`tests/test_prompt_doc.py`가 src의 문자열 상수에 문서 문장이 남아 있지 않은지 검사한다.

현재 문서는 **타입 A**(기준 코드가 만들던 문장을 글자 그대로 옮긴 것)다. 실행 메타(`run_meta.json`의 `prompt_doc`)와
호출 로그(`llm_calls.jsonl`의 `prompt_sha256`, `prompt_sections`)에 문서 경로·SHA-256과 쓴 구간이 남는다.
다른 문서를 쓰려면 설정 `prompts.path`를 바꾼다(교체할 때는 모든 구간을 한 타입으로 통일한다).

## 읽는 법

- 첫 구분선 앞의 `#` 줄(머리 주석)과 `===== 이름 =====` 구분선은 프롬프트에 들어가지 않는다.
- 구간 내용 = 구분선 다음 줄부터 다음 구분선 바로 앞 줄까지를 `\n`으로 이은 것(마지막 줄 뒤 줄바꿈 없음).
  내용의 맨 앞·맨 뒤 줄바꿈은 빈 줄로 보인다. `RESULTS.STATUS_SEPARATOR`(`, `)처럼 끝 공백이 의미 있는 구간이 있으니
  편집기의 끝 공백 자동 삭제를 끈다(테스트가 확인한다).
- `{{이름}}`은 한 번에 치환한다(넣은 값 안의 `{{…}}`는 다시 치환하지 않는다). 값이 없는 자리표시자는 오류.
- 파일 끝은 `===== END =====`.

## 호출별 조립 순서

| 호출 | 메시지 | 조립 |
|---|---|---|
| prep ① 후보 파일 | user | `PREP.FILES_USER`(FILE_LIST = `PREP.FILE_LIST`(HEADER = `PREP.FILE_LIST.HEADER.<단계>`, LIST = 파일 목록; grouped·dirs 단계는 `PREP.FILE_LIST.GROUPED_ITEM`·`DIR_ITEM`), 예산 안에 드는 단계가 없으면 빈 문자열) |
| prep ② 위치 + 재현 테스트 | user | `PREP.LOCATE_USER`(CODE_NOTE = `PREP.CODE_NOTE.<full\|skeleton>`, CODE_CONTEXT = 파일마다 `PREP.CODE_FILE.<full\|skeleton>`을 `PREP.CODE_FILE.SEPARATOR`로 이음; 골격 안의 표시는 `PREP.SKELETON.*`, RUNNER_RULE = `PREP.RUNNER_RULE.<django\|sympy\|pytest>`, V2_BLOCK_PART = 빈 문자열) |
| V2 위치 호출 | user | prep ②와 같고 V2_BLOCK_PART = `V2.BLOCK_PART`(BLOCK = `V2.LAB_TEST` 또는 `V2.LLM_TEST` + `V2.OUTPUT`(NOTE = `V2.OUTPUT_NOTE` 또는 빈 문자열, LOG = 앞 3000자 + `V2.LOG_OMITTED` + 뒤 6000자)). 테스트가 16000자를 넘으면 앞부분 + `V2.TEST_TRUNCATED` |
| 초기 생성 | system `SYSTEM` + user | `INIT_USER`(S2: `INIT_USER.FILE_SCOPE`), LOCATION_CODE = 위치마다 `LOCATION_CODE.ITEM`을 `LOCATION_CODE.SEPARATOR`로 이음(S2: `S2.*`로 만든 코드 문맥), N = 요청 수 |
| 초기 생성 재요청 | system + user | 초기 생성 user(N = 부족분) 바로 뒤에 `INIT_RETRY_NOTICE.HEADER` + 문제 줄들(무효 패치마다 `INIT_RETRY_NOTICE.PATCH_LINE`(WHAT = `INIT_RETRY_NOTICE.<사유>.LINE`, 없으면 `OTHER.LINE`), 이어서 부족 사유 `truncated\|parse_failure.LINE`; 없으면 `NO_PROBLEM_LINES`) + 규칙이 있으면 `RULES_HEADER` + 사유별 `RULE_ITEM`(RULE = `INIT_RETRY_NOTICE.<사유>.RULE`, S2의 search_not_found는 `.RULE.FILE_SCOPE`) |
| 교배 | system + user | 피드백 both: `CROSSOVER_USER.FEEDBACK`, off: `CROSSOVER_USER`. PARENT_A/B = `PARENT.VALID` 또는 `PARENT.INVALID`(LABEL = `PARENT.LABEL_A/B`), RESULTS_A/B = 부모별 실행 결과(아래), SOURCE_PART = `SOURCE_PART`(SOURCE = `RESULTS.SOURCE.APPENDED\|SAVED`) 또는 빈 문자열, EDIT_FORMAT = `EDIT_FORMAT`(S2: `.FILE_SCOPE`) |
| 교배·변이 재요청 | system + user | 직전 user 뒤에 `RETRY_SUFFIX`(DETAIL_PART = `RETRY_SUFFIX.DETAIL_PART` 또는 빈 문자열; search_not_found 세부는 `SEARCH_NOT_FOUND_DETAIL`). 피드백 both면 피드백 항목을 다시 줄여 맞춘다 |
| 변이 | system + user | `MUTATION_USER.FEEDBACK`(off: `MUTATION_USER`, RESULTS = 기존 요약) — 나머지는 교배와 같다 |

부모별 실행 결과(피드백 both, `feedback.exec_summary`): 무효 부모는 `RESULTS.INVALID`. 유효 부모는 줄 단위로
`RESULTS.REPRO`(`RESULTS.PASSED\|FAILED`) → `RESULTS.STATUS_LINE`(항목 `RESULTS.STATUS_ITEM`을 `RESULTS.STATUS_SEPARATOR`로, 20개 초과 시
`RESULTS.STATUS_MORE`) → `RESULTS.OUTPUT_TAIL` → 회귀(`RESULTS.REGRESSION_NONE` | `RESULTS.REGRESSION_NO_FAILURES` |
`RESULTS.REGRESSION_FAILED_COUNT` + 최대 3개 `RESULTS.REGRESSION_FAILURE`) → 시간 초과면 `RESULTS.TIMED_OUT`.
문자 상한으로 자르면 `RESULTS.TRUNCATED`를 붙인다. 회귀 오류 내용 안의 표시는 `REGRESSION_MESSAGE.*`(테스트 파일 프레임, 이름 가림,
프레임 줄). test_files 마스킹 표시는 `MASK.TEST_FILE`.
길이 한도(28,672토큰 추정)를 넘으면 회귀 오류 내용 → 재현 테스트 소스 → 재현 출력(10줄 → 없음) → 테스트별 상태 순으로 뺀다.

## 구간 목록

| 묶음 | 구간 |
|---|---|
| 공통 | `SYSTEM`, `EDIT_FORMAT`, `EDIT_FORMAT.FILE_SCOPE`, `LOCATION_CODE.ITEM`, `LOCATION_CODE.SEPARATOR`, `MASK.TEST_FILE` |
| 초기 생성 | `INIT_USER`, `INIT_USER.FILE_SCOPE`, `INIT_RETRY_NOTICE.{HEADER, NO_PROBLEM_LINES, RULES_HEADER, PATCH_LINE, RULE_ITEM, OTHER.LINE}`, `INIT_RETRY_NOTICE.<사유>.{LINE, RULE}`(사유: search_not_found, match_outside_location, file_not_in_locations, missing_file_path, empty_search, empty_diff, syntax_error, no_edits, truncated, parse_failure, search_not_unique), `INIT_RETRY_NOTICE.search_not_found.RULE.FILE_SCOPE` |
| 교배·변이 | `CROSSOVER_USER`, `CROSSOVER_USER.FEEDBACK`, `MUTATION_USER`, `MUTATION_USER.FEEDBACK`, `SOURCE_PART`, `PARENT.{LABEL_A, LABEL_B, VALID, INVALID, INVALID_EMPTY}`, `RETRY_SUFFIX`, `RETRY_SUFFIX.DETAIL_PART`, `SEARCH_NOT_FOUND_DETAIL` |
| 실행 결과 | `RESULTS.*`(INVALID, REPRO, PASSED, FAILED, STATUS_LINE, STATUS_ITEM, STATUS_SEPARATOR, STATUS_MORE, OUTPUT_TAIL, TIMED_OUT, REGRESSION_NOT_AVAILABLE, REGRESSION_COUNTS, REGRESSION_NONE, REGRESSION_NO_FAILURES, REGRESSION_FAILED_COUNT, REGRESSION_FAILURE, TRUNCATED, SOURCE.APPENDED, SOURCE.SAVED), `REGRESSION_MESSAGE.{TEST_FRAME, NAME_MASK, FRAME, FRAME_CODE}` |
| 전처리 | `PREP.FILES_USER`, `PREP.FILE_LIST`, `PREP.FILE_LIST.HEADER.{full, grouped, dirs, dirs_depth2}`, `PREP.FILE_LIST.{GROUPED_ITEM, DIR_ITEM}`, `PREP.LOCATE_USER`, `PREP.CODE_NOTE.{full, skeleton}`, `PREP.CODE_FILE.{full, skeleton, SEPARATOR}`, `PREP.SKELETON.{END_OF_CLASS, BODY_OMITTED, TRUNCATED}`, `PREP.RUNNER_RULE.{django, sympy, pytest}` |
| V2 위치 호출 | `V2.{BLOCK_PART, LAB_TEST, LLM_TEST, TEST_TRUNCATED, OUTPUT, OUTPUT_NOTE, LOG_OMITTED}` |
| S2 코드 문맥 | `S2.{FULL_FILE, EXCERPT_HEADER, EXCERPT, FILE_SEPARATOR}` |

`RESULTS.REGRESSION_NOT_AVAILABLE`·`REGRESSION_COUNTS`와 `CROSSOVER_USER`·`MUTATION_USER`(피드백 없는 판)는 설정
`loop.feedback: off`일 때만 쓰인다.
