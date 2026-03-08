# Property Repro Platform

물성 예측(특히 LFL) 머신러닝 논문 재현을 위한 자동화 플랫폼입니다.

현재 버전은 다음 흐름을 지원합니다.
- 논문 PDF + 데이터 CSV 업로드
- CSV에서 `SMILES` / `LFL` 컬럼 지정
- PDF를 마크다운으로 변환 (중복 PDF는 캐시 재사용)
- 마크다운 청킹 + 로컬 벡터DB 구성
- `model / feature / hyperparameter / training / metrics` 기준 리트리버 수행
- 리트리버 결과를 최종 비교 보고서용 요약으로 정리

## Tech Stack
- UI: Streamlit
- Agent Orchestration: LangGraph
- LLM: `gpt-5-mini`
- Embedding: `text-embedding-3-large`
- PDF Parsing: `pypdf`
- Vector Store: 로컬 파일 기반 (`vectorstore/`)

## Project Structure
```text
property-repro-platform/
├─ app/                        # Streamlit UI
├─ src/
│  ├─ agents/                  # 에이전트 로직
│  ├─ graph/                   # LangGraph 파이프라인
│  └─ services/                # OpenAI, VectorDB 등 서비스
├─ data/                       # 입력/중간/가공 데이터
├─ artifacts/                  # 생성된 마크다운, 캐시 결과
├─ vectorstore/                # 임베딩/청크 저장소
├─ reports/                    # 최종 보고서 산출물(예정)
└─ requirements.txt
```

## Environment
Anaconda 환경 이름: `paper2property`

`.env` 파일 예시:
```env
OPENAI_API_KEY=your_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

## Run
```powershell
conda activate paper2property
cd C:\Users\bsy\project\property-repro-platform
pip install -r requirements.txt
streamlit run app\app.py
```

## Current Workflow (UI)
1. PDF/CSV 업로드 후 `입력 확정`
2. `PDF 추출 실행`으로 논문 마크다운 생성/재사용
3. `벡터DB 생성 + 멀티 리트리버 실행`으로 리트리버 결과 요약 확인

## Notes
- 코드 생성 단계는 현재 리트리버 요약이 아닌, 저장된 **원문 마크다운**을 기준으로 진행하도록 설계되어 있습니다.
- 동일 PDF는 해시 기반 캐시를 사용해 중복 추출/중복 임베딩을 줄입니다.
