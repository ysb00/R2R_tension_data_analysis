# 작업 기록

`coding_guidelines.md` 9절에 따라 단계마다 기록을 추가한다 (새 작업이 맨 위).

---

## 2026-10-26 — 연구 계획 재작성 (`docs/research_plan.md`)

### 한 일
- 선행 저장소 B(`R2R_steadyState_identification_for_changing_condition`, 로컬 `C:\Users\KIMM\research\informs-paper`)의
  `research_plan.md`, `periodic_disturbance.md`, `hyperparameters.md` §13, `code/03_baselines/evaluation.py` 를 확인했다.
- 최종 목표가 **저속 학습 → 고속 테스트**(B §13.0, 연구자 확정)이고 미학습 장력 조건도 함께 본다는 점을 반영해,
  A 프로젝트의 계획을 `docs/research_plan.md` 로 새로 작성했다.
- 기존 `feature_analysis_plan.md`, `cost_modification_ideas.md` 는 상단에 대체/기록 표시를 달아 남겼다.

### 사용자 승인 (4건)
| 항목 | 확정 |
|---|---|
| 라벨 정책 | label-free 설계 + 설계 동결 후 사후 감사 |
| 속도 분할 경계 | 학습 50/75/100 → 테스트 125~200 (B의 **F1** 결정) |
| 1단계 범위 | P1 거리 좌표계 + P2 진폭 좌표 |
| 조건 축 | 속도축 중심, 장력축은 확인 수준 |

### 판단 근거 (B의 측정값)
- 기존 A 문서는 전부 `Trigger 4` AUROC로 특징을 고르는 구조였고, 이는 B `research_plan.md` 0절
  ("하이퍼파라미터 선택에도 라벨을 사용하지 않는다")에 위반된다. `00_diagnostics.py` 가 같은 이유로 삭제됐다.
- 저속→고속 외삽에서 CoST 가 깨지는 지배적 기전은 진폭이 아니라 **좌표계**다. `BandedFourierLayer` 와
  seasonal loss 가 **절대 주파수 bin 단위**이고 회전 외란이 V=50 0.064 Hz → V=200 0.258 Hz 로 4배 이동한다.
- 진폭 축은 순방향 외삽이다 — 처짐이 −0.23 → −0.46 kg 로 커져 고속 미안정이 저속 기준에서 더 극단적이다.
- 장력 축에서 steady e 표준편차가 0.0379–0.0412 kg 로 평탄하므로, `e ÷ 설정값` 은 오히려 조건 의존성을 만든다.
- 평가 후단이 **시점별** 표현을 받으므로(`evaluation.py`), 특징도 causal 이동창으로 시점마다 산출해야 한다.

### 다음 단계
- `docs/research_plan.md` 5절 승인 대기 12건 검토 → 승인 후 `code/02_rotation_lowspeed.py` 착수.
- A-1 결과(저속 둘레 `C_low`)는 B의 **F4 / G1** 에 그대로 전달된다.
