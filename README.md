# Property Repro Platform

boiling point 회귀 논문 재현을 위한 LangGraph 기반 자동화 플랫폼입니다. 논문 PDF와 분자 데이터를 입력받아, 논문 내용을 마크다운으로 정리하고, RAG로 핵심 실험 설정을 구조화한 뒤, RDKit + scikit-learn 기반 회귀 코드를 생성/실행하고 논문 결과와 비교 보고서를 생성합니다.

## 주요 기능
- 논문 PDF 업로드 및 본문 마크다운 변환
- CSV / Excel 데이터 업로드 및 `SMILES` / 타겟 컬럼 지정
- 마크다운 청킹 + 로컬 벡터 DB 생성
- `model / feature / hyperparameter / training / metrics` 기준 리트리버 수행
- boiling point regression의 단일 best / reported model을 `paper_method_spec` JSON으로 구조화
- `paper_method_spec`를 우선 기준으로 회귀 코드 생성
- RDKit descriptor 또는 Morgan fingerprint 기반 feature 처리
- scikit-learn 회귀 모델 실행 및 `MAE / RMSE / MSE / R2` 계산
- `generate -> run -> debug` 루프 기반 자동 재시도
- 논문 방법론 / 논문 metric / 재현 결과 비교 보고서 생성

## 현재 파이프라인
1. 입력 데이터 설정
   - PDF와 CSV / Excel 업로드
   - `SMILES` 컬럼과 boiling point 타겟 컬럼 지정
2. 논문 파싱
   - `pypdf`로 본문 추출
   - reference, acknowledgements, appendix 등 불필요 섹션 제거
   - 논문 구조를 유지한 마크다운 생성
3. RAG 단계
   - 마크다운을 청크로 분할
   - `text-embedding-3-large`로 임베딩 생성
   - 로컬 벡터 스토어에 저장
   - boiling point regression 중 단일 best / reported model만 선택
   - 결과를 두 형태로 저장
     - 보고서용 markdown table
     - 코드 생성 / 비교용 `paper_method_spec` JSON
4. 코드 생성 및 실행
   - 입력 우선순위
     - `paper_method_spec` (authoritative)
     - `paper_markdown` (background context)
     - `retriever_summary_markdown` (cross-check anchor)
   - RDKit + scikit-learn 기반 실행 가능한 단일 Python 스크립트 생성
   - 실행 실패 시 에러 로그를 반영해 재생성
   - 동일 에러가 반복되면 중단
5. 비교 보고서
   - 논문 기준은 `paper_method_spec`
   - 재현 기준은 자동 디버그 루프의 최종 실행 결과
   - 비교 항목
     - preprocessing
     - feature
     - model
     - hyperparameters
     - training
     - MAE / RMSE / MSE / R2

## 기술 스택
- UI: Streamlit
- Workflow / Agent Orchestration: LangGraph
- LLM: `gpt-5.2`
- Embedding: `text-embedding-3-large`
- PDF Parsing: `pypdf`
- Chemistry Features: `rdkit`
- ML: `scikit-learn`
- Vector Store: 로컬 파일 기반 벡터 저장소

## 프로젝트 구조
```text
property-repro-platform/
├─ AGENTS.md
├─ app/
│  ├─ app.py                     # Streamlit 진입점
│  ├─ sections.py                # UI 섹션 렌더링
│  └─ workflow.py                # UI와 그래프 사이 orchestration
├─ artifacts/
│  ├─ generated_code/
│  ├─ markdown_cache/
│  └─ reports/
├─ data/
│  └─ raw/
├─ src/
│  ├─ agents/
│  │  ├─ code_generation/
│  │  ├─ comparison_report/
│  │  ├─ code_execution_agent.py
│  │  ├─ code_generation_agent.py
│  │  ├─ code_loop_agent.py
│  │  ├─ paper_parsing_agent.py
│  │  └─ retriever_table_agent.py
│  ├─ graph/
│  │  ├─ code_execution_graph.py
│  │  ├─ code_generation_graph.py
│  │  ├─ code_loop_graph.py
│  │  ├─ comparison_report_graph.py
│  │  ├─ paper_parsing_graph.py
│  │  └─ rag_graph.py
│  ├─ services/
│  │  └─ vector_db_service.py
│  └─ utils/
│     ├─ code_text.py
│     ├─ openai_client.py
│     └─ paper_method_spec.py
├─ vectorstore/
├─ README.md
└─ requirements.txt
```

## 핵심 데이터 계약
### `paper_method_spec`
section 6의 구조화 결과로, 이후 단계의 단일 truth source로 사용합니다.

예시 필드:
```json
{
  "preprocessing": "drop invalid SMILES, remove duplicates, impute missing values",
  "feature": {
    "method": "rdkit_descriptor",
    "details": "use RDKit molecular descriptors"
  },
  "model": {
    "name": "random_forest",
    "display_name": "RandomForestRegressor"
  },
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42
  },
  "training": {
    "split": "train_test_split",
    "test_size": 0.2,
    "random_state": 42
  },
  "metrics": {
    "MAE": 12.3,
    "RMSE": 18.7,
    "MSE": 349.7,
    "R2": 0.88
  },
  "selection_basis": "best reported boiling point regression model in the paper"
}
```

## 실행 환경
Anaconda 환경 이름: `paper2property`

예시 `.env`:
```env
OPENAI_API_KEY=your_api_key
OPENAI_CHAT_MODEL=gpt-5.2
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

## 실행 방법
```powershell
conda activate paper2property
cd C:\Users\bsy\project\property-repro-platform
pip install -r requirements.txt
streamlit run app\app.py
```

## UI 사용 순서
1. PDF와 CSV / Excel 업로드
2. `입력 확정`
3. `PDF 추출 실행`
4. `벡터DB 생성 + 멀티 리트리버 실행`
5. `코드 생성 및 실행 시작`
6. `비교 보고서 생성`

## 생성 코드 기본 정책
- chemistry feature는 RDKit 사용
- 모델링은 scikit-learn 사용
- metric은 `MAE / RMSE / MSE / R2`를 모두 계산
- 논문 정보가 누락되면 필요한 최소 기본값만 보완
- feature에서 제외하는 컬럼
  - `SMILES`
  - target column
  - compound / molecule / id / InChI / CAS 계열 식별 컬럼

## 디버그 루프 동작
- 최대 반복 횟수 내에서 `generate -> run -> debug` 수행
- 실패 시 `stderr`, `stdout`, `error`를 모두 저장
- 최근 에러 이력을 다음 코드 생성 프롬프트에 반영
- 동일한 정규화 에러 서명이 2회 이상 반복되면 중단
- 최종 결과에는 아래가 포함됨
  - `generated_code`
  - `execution_result`
  - `error_history`
  - `latest_error`
  - `latest_stdout`
  - `latest_stderr`
  - `stop_reason`

## 산출물
- `artifacts/markdown_cache/`: PDF별 마크다운 캐시
- `vectorstore/`: 청크 / 임베딩 / 메타데이터
- `artifacts/generated_code/`: 생성된 Python 코드
- `artifacts/reports/`: 논문 대비 재현 결과 비교 보고서

## 제한 사항
- 현재 범위는 boiling point regression 재현에 맞춰져 있습니다.
- 논문 구조가 매우 불규칙하거나 본문 OCR 품질이 낮으면 structured spec 품질이 떨어질 수 있습니다.
- 완전한 논문 재현보다는, 논문에 보고된 핵심 방법론을 최대한 보존한 실행 가능한 baseline 재현에 가깝습니다.
