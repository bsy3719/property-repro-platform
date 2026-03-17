# Property Repro Platform

화학/물성 예측 논문의 회귀 실험을 자동으로 재현하는 LangGraph 기반 플랫폼입니다.  
현재는 `boiling point` 회귀를 기본 대상으로 하며, 논문 PDF와 분자 데이터를 입력받아 다음 흐름을 수행합니다.

1. 논문 본문을 구조 보존형 Markdown으로 변환
2. RAG로 모델/feature/학습 조건을 추출해 `paper_method_spec` 생성
3. RDKit + scikit-learn 기반 재현 코드를 자동 생성
4. 코드를 실행하고 `MAE`, `RMSE`, `MSE`, `R2`를 수집
5. 논문 보고값과 재현값을 비교하는 보고서 생성

## 핵심 기능

- PDF + CSV/Excel 업로드 기반 재현 워크플로우
- `SMILES` / target 컬럼 자동 추천
- 논문 Markdown 캐싱과 로컬 vector store 기반 검색
- `paper_method_spec` 중심의 구조화 추출
- RDKit chemistry-aware 코드 생성
- `generate -> run -> debug` 자동 복구 루프
- 논문 대비 재현 결과 비교 보고서 생성

## 아키텍처 개요

### 입력에서 결과까지

1. `ColumnSelectionAgent`
   - 업로드된 표 데이터에서 `SMILES` 컬럼과 target 컬럼을 추천
2. `PaperParsingAgent`
   - PDF 텍스트를 구조 보존형 Markdown으로 변환
3. RAG 파이프라인
   - Markdown 청킹
   - 임베딩 생성
   - vector store 검색
   - evidence 수집
4. `ModelSelectionAgent`, `FeatureEvidenceAgent`, `MethodSectionAgent`
   - 최종 모델, feature, 전처리/학습 조건/metric을 구조화
5. `RetrieverTableAgent`
   - 결과를 `paper_method_spec`으로 정규화
6. `CodeGenerationAgent`
   - `paper_method_spec` 기반 Python 재현 코드 생성
7. `CodeExecutionAgent`
   - 코드 저장, 실행, metric 파싱
8. `CodeGenerationRunDebugAgent`
   - 실행 실패 시 자동 수정 루프 수행
9. `ComparisonReportAgent`
   - 논문 결과와 재현 결과 차이를 보고서로 생성

## 기술 스택

- Agent orchestration: `LangGraph`
- API / UI: `FastAPI` + HTML
- LLM: `gpt-5.2`
- Embedding: `text-embedding-3-large`
- PDF parsing: `pypdf`
- Chemistry: `RDKit`
- ML: `scikit-learn`
- Vector DB: 로컬 `vectorstore/`

## 프로젝트 구조

```text
property-repro-platform/
├─ app/
│  ├─ backend_core.py
│  ├─ chem_repro_platform.html
│  └─ fastapi_server.py
├─ docs/
├─ reference/
├─ src/
│  ├─ agents/
│  │  ├─ code_generation/
│  │  ├─ code_verification/
│  │  ├─ comparison_report/
│  │  ├─ code_execution_agent.py
│  │  ├─ code_generation_agent.py
│  │  ├─ code_loop_agent.py
│  │  ├─ column_selection_agent.py
│  │  ├─ feature_evidence_agent.py
│  │  ├─ method_section_agent.py
│  │  ├─ model_selection_agent.py
│  │  ├─ paper_parsing_agent.py
│  │  └─ retriever_table_agent.py
│  ├─ graph/
│  ├─ services/
│  └─ utils/
├─ tests/
├─ AGENTS.md
├─ README.md
└─ requirements.txt
```

## 핵심 데이터 계약

### `paper_method_spec`

RAG 단계의 최종 구조화 산출물이며, 코드 생성의 주 입력으로 사용됩니다.

대표 필드:

```json
{
  "feature": {
    "method": "rdkit_descriptor",
    "descriptor_names": ["MolWt", "TPSA"],
    "count_feature_names": ["HeavyAtomCount"]
  },
  "model": {
    "name": "RandomForestRegressor"
  },
  "hyperparameters": {
    "n_estimators": 300,
    "random_state": 42
  },
  "training": {
    "split_strategy": "train_test_split",
    "test_size": 0.2
  },
  "metrics": {
    "reported": {
      "MAE": 19.876,
      "RMSE": 43.295,
      "MSE": 1874.417,
      "R2": 0.881
    }
  }
}
```

## 설치

권장 환경은 conda `paper2property`입니다.

```bash
conda create -n paper2property python=3.11 -y
conda activate paper2property
pip install -r requirements.txt
```

`.env` 예시:

```env
OPENAI_API_KEY=your_api_key
OPENAI_CHAT_MODEL=gpt-5.2
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

## 실행

FastAPI + HTML UI:

```bash
conda run -n paper2property python app/fastapi_server.py
```

접속 주소:

```text
http://localhost:8790/
```

## 테스트

전체 테스트:

```bash
conda run -n paper2property python -m pytest tests/
```

빠른 문법 확인:

```bash
conda run -n paper2property python -m compileall app src tests
```

## 사용 흐름

1. 논문 PDF와 CSV/Excel 파일 업로드
2. 추천된 `SMILES` / target 컬럼 확인
3. 논문 파싱 실행
4. RAG 실행
5. 코드 생성 및 실행
6. 비교 보고서 확인

## 생성 코드 정책

- 입력의 단일 source of truth는 `paper_method_spec`
- chemistry feature는 `RDKit` 기반으로 계산
- 모델링은 `scikit-learn` 사용
- 출력 metric은 항상 `MAE`, `RMSE`, `MSE`, `R2`
- 논문에 없는 세부값은 `assumptions`에 기록
- 지원되지 않는 외부 descriptor 도구는 임의 치환하지 않음

## 디버그 루프

- 기본 흐름은 `generate -> run -> debug`
- 같은 오류가 2회 이상 반복되면 중단
- 최대 반복 횟수는 제한됨
- 필요 시 safety-net 코드로 마무리

최종 결과에는 보통 다음이 포함됩니다.

- `generated_code`
- `execution_result`
- `assumptions`
- `error_history`
- `stop_reason`

## 생성 산출물

아래 경로들은 실행 중 자동 생성되며 `.gitignore`에 포함됩니다.

- `artifacts/markdown_cache/`
- `artifacts/generated_code/`
- `artifacts/reports/`
- `artifacts/results/`
- `vectorstore/`
- `data/raw/`

## 현재 범위와 제한

- 현재 기본 대상은 `boiling point` 회귀 재현입니다.
- 논문에 핵심 설정이 누락된 경우 일부 기본값이 사용될 수 있습니다.
- OCR 품질이 낮거나 표/식이 심하게 깨진 PDF는 추출 품질이 떨어질 수 있습니다.
- unsupported external descriptors(Mordred, PaDEL, Dragon 등)는 직접 재현하지 않습니다.

## Git 업로드 전 체크

- `.env`가 커밋 대상에 포함되지 않았는지 확인
- `artifacts/`, `vectorstore/`, `data/raw/`가 제외되는지 확인
- 로컬 에이전트 설정 파일(`.claude/`, `CLAUDE.md`)이 제외되는지 확인
- 테스트 통과 여부 확인

```bash
git status --short
conda run -n paper2property python -m pytest tests/
```
