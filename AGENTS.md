# AGENTS.md

## Project
- Name: `property-repro-platform`
- Root: `/home/bsy/project/property-repro-platform`
- Purpose: reproduce machine learning experiments from chemistry/property prediction papers and compare reproduced regression metrics with the paper
- Current target property: `boiling point`
- Main framework: `LangGraph`
- Default chat model: `gpt-5.2`
- Default embedding model: `text-embedding-3-large`

## Environment
- Conda env: `paper2property`
- Important env vars:
  - `.env` contains `OPENAI_API_KEY`
  - optional `OPENAI_CHAT_MODEL`
  - optional `OPENAI_EMBEDDING_MODEL`
- Common Python path:
  - `/home/bsy/miniconda3/envs/paper2property/bin/python`

## Run Commands
- FastAPI + HTML UI:
  - `conda run -n paper2property python app/fastapi_server.py`
  - serves on `http://localhost:8790/`
- Tests:
  - `conda run -n paper2property python -m pytest tests/`
- Quick syntax check:
  - `conda run -n paper2property python -m compileall app src tests`

## Current UI
### FastAPI + HTML
- Shared backend: `app/backend_core.py`
- API server: `app/fastapi_server.py`
- HTML client: `app/chem_repro_platform.html`

## Current Pipeline
1. Upload paper PDF and tabular data (`csv`, `xlsx`, `xls`)
2. Detect `SMILES` column and target column with `ColumnSelectionAgent`
3. Allow user confirmation of selected columns
4. Parse PDF with `pypdf` and convert to markdown while preserving useful paper structure
5. Cache markdown by PDF hash in `artifacts/markdown_cache`
6. Build local vector DB from markdown chunks using OpenAI embeddings
7. Retrieve `model / feature / hyperparameter / training / metrics` with chemistry-aware multi-query RAG
8. Convert retrieval output into structured `paper_method_spec`
9. Generate executable Python regression code with LangGraph
10. Execute generated code with LangGraph
11. Run `generate -> run -> debug` loop when execution fails
12. Optionally compare reproduced metrics with paper metrics

## Input Policy For Code Generation
- Main source of truth: structured `paper_method_spec`
- Background context: full paper markdown
- Required behavior:
  - use structured spec as the primary implementation contract
  - use markdown only as background context outside code generation when needed
  - if markdown and spec conflict, prefer the final structured spec for selected model details
- Goal of generated code:
  - run the machine learning model described in the paper
  - obtain regression metrics from the reproduced run

## Chemistry-Aware Code Generation Rules
- Generate raw Python code only
- Never return markdown fences like ```` ```python ````
- Use `RDKit` for SMILES featurization
- Use `scikit-learn` for modeling
- Always include metrics:
  - `MAE`
  - `RMSE`
  - `MSE`
  - `R2`
- Use the canonical scaffold expected by validation:
  - `SPEC`
  - `load_dataframe`
  - `mol_from_smiles`
  - `build_descriptor_matrix`
  - `build_count_feature_matrix`
  - `build_fingerprint_matrix`
  - `assemble_feature_matrix`
  - `build_model`
  - `train_and_evaluate`
  - `main`
- Prefer explicit chemistry features from the paper over full-descriptor fallback
- Supported chemistry feature families currently focus on `RDKit only`
- Handle named RDKit descriptors, count features, and fingerprints such as:
  - descriptor subsets from `Descriptors.descList`
  - count features like `HeavyAtomCount`, `RingCount`, `NumRotatableBonds`, `NHOHCount`, `NOCount`
  - Morgan/ECFP, MACCS, atom-pair, topological torsion, RDKit fingerprints
- If unsupported external descriptor tools such as Mordred/PaDEL/Dragon are mentioned without implementable detail:
  - keep them as warnings/assumptions
  - do not silently map them to different chemistry features

## Central Structured Contract
### `paper_method_spec`
- Main structured output of the RAG stage
- Used as the authoritative input for code generation and reporting
- Important `feature` fields now include:
  - `method`
  - `descriptor_names`
  - `count_feature_names`
  - `fingerprint_family`
  - `radius`
  - `n_bits`
  - `feature_terms`
  - `unresolved_feature_terms`

## Implemented Agents
### `src/agents/paper_parsing_agent.py`
- Parses extracted PDF text into markdown
- Preserves paper structure
- Filters references/appendix-like sections

### `src/agents/retriever_table_agent.py`
- Assembles retrieved paper evidence into normalized `paper_method_spec`
- Produces report-friendly summaries derived from the structured spec

### `src/agents/column_selection_agent.py`
- LLM-based detection of `SMILES` and target columns from uploaded tabular data

### `src/agents/code_generation/`
- Main code generation package
- Important modules:
  - `agent.py`
  - `prompting.py`
  - `validation.py`
  - `normalization.py`
  - `defaults.py`
  - `fallback_script.py`
  - `few_shot_examples.py`
- Includes chemistry-aware prompt construction, normalization, fallback generation, and validation

### `src/agents/code_execution_agent.py`
- Saves generated code if needed
- Builds execution command
- Runs the generated script
- Parses JSON-like stdout
- Returns metrics, stdout, stderr, return code

### `src/agents/code_loop_agent.py`
- LangGraph loop agent
- Flow: `generate -> run -> debug`
- Default max iteration used in UI: `4`
- Hard clamp: `3~5`
- Stops if the same error repeats at least 2 times

### `src/agents/comparison_report/`
- Builds paper-vs-reproduced metric comparison reports

## Implemented Graph Entrypoints
- `src/graph/paper_parsing_graph.py`
- `src/graph/rag_graph.py`
- `src/graph/code_generation_graph.py`
- `src/graph/code_execution_graph.py`
- `src/graph/code_loop_graph.py`
- `src/graph/comparison_report_graph.py`

## Retrieval / Feature Notes
- `src/graph/rag_graph.py` uses multi-query retrieval for chemistry feature discovery
- Feature retrieval now expands across terms like:
  - descriptor
  - physicochemical descriptor
  - RDKit descriptor
  - Morgan / ECFP / MACCS
  - atom count / bond count / ring count
  - graph / adjacency / bond matrix
- `src/utils/chemistry_features.py` centralizes descriptor snapshots and alias normalization

## Storage / Caching
- Raw uploads: `data/raw`
- Markdown artifacts: `artifacts/`
- Markdown cache by PDF hash: `artifacts/markdown_cache`
- Generated code: `artifacts/generated_code`
- Comparison reports: `artifacts/reports`
- Local vector DB: `vectorstore/`

## Known Technical Notes
- FastAPI sessions are currently stored in memory in `app/fastapi_server.py`
- HTML UI is step-based and calls the backend through `/api/upload`, `/api/select-sheet`, `/api/confirm-columns`, `/api/parse-paper`, `/api/run-rag`, `/api/generate`
- `pypdf` is required for paper parsing
- `python-multipart` is required for FastAPI file upload handling
- `matplotlib` may still warn about cache directory permissions depending on environment

## Installed Packages Confirmed Recently
- `rdkit`
- `scikit-learn`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `pandas`
- `pypdf`
- `fastapi`
- `uvicorn`
- `python-multipart`

## Current Conventions
- Do not ask before normal implementation work; execute directly unless there is hidden risk
- Prefer editing files directly over relying on nonexistent init helpers
- Keep responses concise
- For this project, use boiling point regression assumptions unless explicitly changed
- When working on chemistry feature extraction/codegen, prefer explicit descriptor/count/fingerprint evidence over generic full-descriptor fallback

## Validation / Testing Status
- Unit tests exist in `tests/`
- Recent chemistry-aware code generation tests live under `tests/test_chemistry_code_generation.py`
- End-to-end validation still requires running the UI and actual model/API calls

## Next Likely Tasks
- Connect comparison report generation into the FastAPI/HTML flow
- Add persistent session storage for FastAPI instead of in-memory sessions
- Improve end-to-end validation for the alternative HTML UI
- Export structured run artifacts for report/debug review
