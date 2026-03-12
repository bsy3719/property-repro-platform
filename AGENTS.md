# AGENTS.md

## Project
- Name: `property-repro-platform`
- Root: `C:\Users\bsy\project\property-repro-platform`
- Purpose: reproduce machine learning experiments from chemistry/property prediction papers and compare reproduced regression metrics with the paper.
- Current target property: `boiling point`
- Current UI: `Streamlit`
- Agent framework: `LangGraph`
- LLM model default: `gpt-5.2`
- Embedding model: `text-embedding-3-large`

## Environment
- Conda env: `paper2property`
- Important env var: `.env` contains `OPENAI_API_KEY`
- Main runtime command:
  - `streamlit run app\app.py`
- Python path often used:
  - `C:\Users\bsy\anaconda3\envs\paper2property\python.exe`

## Current Pipeline
1. Upload paper PDF and tabular data (`csv`, `xlsx`, `xls`) in Streamlit.
2. Select `SMILES` column and target column for `boiling point`.
3. Parse PDF with `pypdf` and convert to markdown while preserving paper structure and filtering useless sections such as references.
4. Cache markdown by PDF hash so the same PDF is not parsed repeatedly.
5. Build local vector DB from markdown chunks using OpenAI embeddings.
6. Retrieve `model / feature / hyperparameter / training / metrics` for boiling point regression.
7. Summarize retrieval results for reporting.
8. Generate executable Python regression code with LangGraph.
9. Execute generated code with LangGraph.
10. Run `generate -> run -> debug` loop with LangGraph when execution fails.

## Input Policy For Code Generation
- Main source of truth: full paper markdown.
- Anchor source: retriever/model summary table.
- Required behavior:
  - use full markdown for implementation context.
  - use model summary as the anchor for the selected best model/configuration.
  - if markdown and summary conflict, prefer summary for the final selected model, but use markdown for surrounding implementation details.
- Goal of generated code:
  - run the machine learning model described in the paper.
  - obtain regression metrics from the reproduced run.

## Code Generation Rules
- Generate raw Python code only.
- Never return markdown fences like ```` ```python ````.
- Use `RDKit` for SMILES featurization.
- Use `sklearn` for modeling.
- Always include metrics:
  - `MAE`
  - `RMSE`
  - `MSE`
  - `R2`
- Handle missing/incomplete paper details with reasonable defaults.
- Default preprocessing fallback includes:
  - invalid SMILES drop
  - missing target drop
  - duplicate removal
  - median imputation when needed
  - scaling when model type requires it
- Do not use identifier-like columns as model features.
- Exclude these from additional tabular features:
  - `smiles`
  - target column
  - compound/name/id/inchi/cas-like columns
- Use SMILES only for RDKit feature generation.

## Implemented Agents
### `src/agents/paper_parsing_agent.py`
- Parses extracted PDF text into markdown.
- Preserves paper structure.
- Filters references/appendix-like sections.

### `src/agents/retriever_table_agent.py`
- Summarizes retrieved paper evidence.
- Current purpose: report-oriented summary for boiling point regression.

### `src/agents/code_generation_agent.py`
- LangGraph-based code generation agent.
- Inputs:
  - `paper_markdown`
  - `model_anchor_summary`
  - dataset info
- Produces executable Python regression code.
- Sanitizes markdown fences if model output includes them.
- Validates presence of rdkit/sklearn/metrics and Python syntax.

### `src/agents/code_execution_agent.py`
- Saves generated code to file if needed.
- Builds execution command.
- Runs the generated script.
- Parses JSON-like stdout.
- Returns metrics, stdout, stderr, return code.

### `src/agents/code_loop_agent.py`
- LangGraph loop agent.
- Flow: `generate -> run -> debug`.
- Default max iteration used in UI: `4`.
- Hard clamp: `3~5`.
- Stops if same error repeats at least 2 times.

## Implemented Graph Entrypoints
- `src/graph/paper_parsing_graph.py`
- `src/graph/rag_graph.py`
- `src/graph/code_generation_graph.py`
- `src/graph/code_execution_graph.py`
- `src/graph/code_loop_graph.py`

## UI Structure
Main file:
- `app/app.py`

Current Streamlit sections:
1. PDF upload
2. Data upload
3. Column selection
4. Preview
5. PDF parsing and markdown display
6. Vector DB + retriever summary
7. Code generation
8. Code execution / auto-debug loop

Important UI behavior:
- Generated Python code should appear only once.
- There is a direct execution button.
- There is an auto-debug execution button.

## Storage / Caching
- Raw uploads: `data/raw`
- Markdown artifacts: `artifacts/`
- Markdown cache by PDF hash: `artifacts/markdown_cache`
- Generated code: `artifacts/generated_code`
- Local vector DB: `vectorstore/`

## Known Technical Notes
- Streamlit server sometimes exits if launched in-process from the session.
- More stable launch method has been using a separate PowerShell window.
- `matplotlib` import works, but may warn about cache directory permissions under `C:\Users\bsy\.matplotlib`.
- Numerical stack in `paper2property` was repaired after `numpy/scipy/sklearn` conflicts.

## Installed Packages Confirmed Recently
- `rdkit`
- `scikit-learn`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `pandas 2.3.3`

## Current Conventions
- Do not ask before normal implementation work; execute directly unless there is hidden risk.
- Prefer editing files directly over relying on nonexistent init helpers.
- Keep responses concise.
- For this project, use boiling point regression assumptions unless explicitly changed.

## Next Likely Tasks
- Generate final comparison report between paper metrics and reproduced metrics.
- Improve retriever summary format for report generation.
- Connect execution results to reporting agent.
- Optionally stabilize matplotlib cache path with `MPLCONFIGDIR`.
- Optionally add structured export of code-generation/report artifacts.
