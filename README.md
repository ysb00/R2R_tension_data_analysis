# R2R_tension_data_analysis

for R2R_steadyState_identification

장력 신호의 **transient / steady-state 식별력 feature 사전 분석**. 선행 저장소
`R2R_steadyState_identification_for_changing_condition`(이하 B)에서 CoST 가 일관 1위였으므로,
제안 방법은 CoST 변형으로 가고 본 저장소는 **그 변형에 넣을 입력 feature 를 결정**한다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/research_plan.md](docs/research_plan.md) | **유효 계획.** 범위·승계 규약·feature 후보·분석 설계·한계. 8절에 실행 후 정정 사항 |
| [docs/outputs_guide.md](docs/outputs_guide.md) | 산출물 안내 — csv 컬럼과 plot 축·목적·해석 |
| [docs/worklog.md](docs/worklog.md) | 단계별 작업 기록 (새 작업이 맨 위) |
| [docs/coding_guidelines.md](docs/coding_guidelines.md) | 코드 작성 규칙 |
| [docs/data_dictionary.md](docs/data_dictionary.md) | 원본 컬럼 설명 |
| `docs/reference_*.md` | 이전 계획 문서 4개 (참고용 보존. 그 전제는 현 계획에 승계되지 않는다) |

## 실행

```bash
python code/01_merge.py         # 원본 zip -> outputs/01_merge/merged_data.csv (재생성 불필요)
python code/02_features.py      # 시점별 feature 16종 x 창 3종 (라벨 미사용)
python code/03_screening.py     # D·T, 승계 검증, 증분·전진 선택, 창 민감도, 진폭 정규화 (약 22분)
python code/04_overlap.py       # 상태 정보 vs 조건 정보의 중첩
python code/05_visualize.py     # spectrogram, phase portrait
```

`code/evaluation.py` 와 `code/splits.py` 는 B 저장소에서 **수정 없이 복사**한 평가 후단이다.
출처와 커밋 hash 는 [code/external_versions.csv](code/external_versions.csv) 에 기록돼 있다.
`Trigger 4` 는 `evaluation.py` 의 `_labels()` 한 곳에서만 읽힌다 (라벨 누출 차단).

## 결과물 폴더

단계마다 **제안 4계열 기준**으로 나뉜다 (`outputs/NN_단계/` 아래).

| 폴더 | feature | 비고 |
|---|---|---|
| `P1_spectral/` | spectral centroid · roll-off · bandwidth | 제안 1 |
| `P2_phase/` | instantaneous phase velocity | 제안 2 (PLV 기각) |
| `P3_nonstationarity/` | rolling kurtosis · skewness · Hurst | 제안 3 |
| `P4_physical/` | Hilbert envelope mean · std | 제안 4 (Tension-Speed 상관 기각) |
| `P0_baseline/` | moving std · slope | B 통계 baseline 승계 기준점 |
| `P5_additional/` | 진폭·추세 계열 5종 | 계획 3.2절에서 추가 |
| `_cross/` | 계열을 넘나드는 결과와 통합본 | |
