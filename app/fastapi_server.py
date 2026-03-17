from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from .backend_core import (
        build_session_from_upload,
        detect_columns_with_llm,
        load_tabular_path,
        parse_paper_for_session,
        run_generation_for_session,
        run_rag_for_session,
    )
except ImportError:
    from backend_core import (
        build_session_from_upload,
        detect_columns_with_llm,
        load_tabular_path,
        parse_paper_for_session,
        run_generation_for_session,
        run_rag_for_session,
    )

APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "chem_repro_platform.html"
RESULTS_DIR = APP_DIR.parent / "artifacts" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS: dict[str, dict[str, Any]] = {}


class SessionPayload(BaseModel):
    session_id: str


class SheetPayload(SessionPayload):
    sheet_name: str | None = None


class ColumnPayload(SessionPayload):
    smiles_column: str | None = None
    target_column: str | None = None


class RagPayload(SessionPayload):
    top_k: int = 5


class SaveResultPayload(BaseModel):
    session_id: str
    final_output: dict[str, Any]


app = FastAPI(title="Paper2Property HTML API", version="0.1.0")


def get_session_or_404(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return session


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_existing_upload_session(
    sessions: dict[str, dict[str, Any]],
    pdf_hash: str,
    data_hash: str,
    sheet_name: str | None = None,
) -> dict[str, Any] | None:
    normalized_sheet_name = (sheet_name or "").strip() or None
    for session in sessions.values():
        if session.get("pdf_hash") != pdf_hash:
            continue
        if session.get("data_hash") != data_hash:
            continue
        if session.get("data_sheet_name") != normalized_sheet_name:
            continue
        return session
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_metric_dict(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in ["MAE", "RMSE", "MSE", "R2"]:
        value = metrics.get(key)
        if value is not None:
            normalized[key] = _json_safe(value)
    return normalized


def _build_result_id(session: dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_hash = str(session.get("pdf_hash", "unknown"))[:8]
    data_hash = str(session.get("data_hash", "unknown"))[:8]
    base_id = f"result_{timestamp}_{pdf_hash}_{data_hash}"
    candidate = base_id
    suffix = 1
    while (RESULTS_DIR / f"{candidate}.json").exists():
        suffix += 1
        candidate = f"{base_id}_{suffix}"
    return candidate


def build_result_record(session: dict[str, Any], final_output: dict[str, Any]) -> dict[str, Any]:
    execution_result = final_output.get("execution_result", {}) if isinstance(final_output, dict) else {}
    parsed_output = execution_result.get("parsed_output", {}) if isinstance(execution_result, dict) else {}
    reproduction_summary = final_output.get("reproduction_summary", {}) if isinstance(final_output, dict) else {}
    comparison_report = final_output.get("comparison_report", {}) if isinstance(final_output, dict) else {}
    paper_method_spec = session.get("paper_method_spec", {})

    paper_metrics = _normalize_metric_dict(
        reproduction_summary.get("paper_metrics")
        or parsed_output.get("paper_reported_metrics")
        or (paper_method_spec.get("metrics", {}).get("reported", {}) if isinstance(paper_method_spec, dict) else {})
        or {}
    )
    reproduced_metrics = _normalize_metric_dict(
        reproduction_summary.get("reproduced_metrics")
        or parsed_output.get("metrics")
        or execution_result.get("metrics")
        or {}
    )

    status = str(reproduction_summary.get("status", "")).strip().lower() or "failed"
    headline = str(reproduction_summary.get("headline", "")).strip()
    paragraphs = [str(p).strip() for p in reproduction_summary.get("paragraphs", []) if str(p).strip()]

    raw_assumptions = parsed_output.get("assumptions", [])
    assumptions = [str(item).strip() for item in raw_assumptions if str(item).strip()] if isinstance(raw_assumptions, list) else []

    comparison_report_markdown = ""
    if isinstance(comparison_report, dict):
        comparison_report_markdown = (
            str(comparison_report.get("report_markdown", "")).strip()
            or str(comparison_report.get("comparison_table_markdown", "")).strip()
        )

    model_name = ""
    feature_method = ""
    if isinstance(paper_method_spec, dict):
        model_name = str(paper_method_spec.get("model", {}).get("name", "") or "")
        feature_method = str(paper_method_spec.get("feature", {}).get("method", "") or "")

    return {
        "result_id": _build_result_id(session),
        "saved_at": datetime.now().isoformat(),
        "pdf_name": str(session.get("pdf_name", "") or ""),
        "data_name": str(session.get("data_name", "") or ""),
        "model_name": model_name,
        "feature_method": feature_method,
        "paper_metrics": paper_metrics,
        "reproduced_metrics": reproduced_metrics,
        "reproduction_status": status,
        "reproduction_headline": headline,
        "reproduction_paragraphs": paragraphs,
        "assumptions": assumptions,
        "paper_method_spec": _json_safe(paper_method_spec),
        "comparison_report_markdown": comparison_report_markdown,
    }


@app.get("/")
@app.get("/chem-repro")
async def serve_app() -> FileResponse:
    return FileResponse(HTML_PATH, media_type="text/html")


@app.get("/api/session")
async def get_session(id: str) -> dict[str, Any]:
    return {"session": get_session_or_404(id)}


@app.post("/api/upload")
async def upload_files(
    pdf_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    sheet_name: str | None = Form(default=None),
) -> dict[str, Any]:
    pdf_bytes = await pdf_file.read()
    data_bytes = await data_file.read()
    if not pdf_bytes or not data_bytes:
        raise HTTPException(status_code=400, detail="pdf_file과 data_file을 함께 업로드해야 합니다.")

    normalized_sheet_name = (sheet_name or "").strip() or None
    existing_session = find_existing_upload_session(
        SESSIONS,
        pdf_hash=_sha256_bytes(pdf_bytes),
        data_hash=_sha256_bytes(data_bytes),
        sheet_name=normalized_sheet_name,
    )
    if existing_session:
        return {
            "session": existing_session,
            "duplicate_upload": True,
            "message": "동일한 논문/데이터 파일이 이미 업로드되어 기존 세션을 재사용합니다.",
        }

    try:
        session = build_session_from_upload(
            pdf_name=pdf_file.filename or "paper.pdf",
            pdf_bytes=pdf_bytes,
            data_name=data_file.filename or "data.csv",
            data_bytes=data_bytes,
            sheet_name=normalized_sheet_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    SESSIONS[session["session_id"]] = session
    return {"session": session, "duplicate_upload": False}


@app.post("/api/select-sheet")
async def select_sheet(payload: SheetPayload) -> dict[str, Any]:
    session = get_session_or_404(payload.session_id)
    try:
        dataframe, selected_sheet, _, available_sheets = load_tabular_path(
            session["data_path"],
            sheet_name=payload.sheet_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    session["data_sheet_name"] = selected_sheet
    session["available_sheets"] = available_sheets
    session["data_columns"] = dataframe.columns.tolist()
    session["num_rows"] = int(len(dataframe))
    session["num_columns"] = int(len(dataframe.columns))
    preview = dataframe.head(8)
    session["data_preview"] = preview.where(preview.notna(), None).to_dict(orient="records")
    session["column_detection"] = detect_columns_with_llm(
        dataframe,
        file_hash=session["data_hash"],
        selected_sheet=selected_sheet,
    )
    if session["data_columns"]:
        session["smiles_column"] = session["column_detection"].get("smiles_column") or session["data_columns"][0]
        session["target_column"] = session["column_detection"].get("target_column") or session["data_columns"][0]
    return {"session": session}


@app.post("/api/confirm-columns")
async def confirm_columns(payload: ColumnPayload) -> dict[str, Any]:
    session = get_session_or_404(payload.session_id)
    session["smiles_column"] = payload.smiles_column or session.get("smiles_column")
    session["target_column"] = payload.target_column or session.get("target_column")
    if session["smiles_column"] == session["target_column"]:
        raise HTTPException(status_code=400, detail="SMILES 컬럼과 타겟 컬럼은 서로 달라야 합니다.")
    return {"session": session}


@app.post("/api/parse-paper")
async def parse_paper(payload: SessionPayload) -> dict[str, Any]:
    session = get_session_or_404(payload.session_id)
    try:
        result = parse_paper_for_session(session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"result": result, "session": session}


@app.post("/api/run-rag")
async def run_rag(payload: RagPayload) -> dict[str, Any]:
    session = get_session_or_404(payload.session_id)
    try:
        result = run_rag_for_session(session, top_k=int(payload.top_k))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"result": result, "session": session}


@app.post("/api/generate")
async def generate(payload: SessionPayload) -> dict[str, Any]:
    session = get_session_or_404(payload.session_id)
    try:
        result = run_generation_for_session(session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"result": result, "session": session}


@app.post("/api/save-result")
async def save_result(payload: SaveResultPayload) -> dict[str, Any]:
    session = get_session_or_404(payload.session_id)
    final_output = payload.final_output if isinstance(payload.final_output, dict) else {}
    record = build_result_record(session, final_output)
    result_id = record["result_id"]
    file_path = RESULTS_DIR / f"{result_id}.json"
    file_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "result_id": result_id, "path": str(file_path)}


@app.get("/api/results")
async def list_results() -> dict[str, Any]:
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            results.append(record)
        except Exception:
            continue
    return {"results": results}


@app.delete("/api/results/{result_id}")
async def delete_result(result_id: str) -> dict[str, Any]:
    if not result_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid result_id")
    file_path = RESULTS_DIR / f"{result_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    file_path.unlink()
    return {"ok": True, "result_id": result_id}


def run_server(host: str = "0.0.0.0", port: int = 8790) -> None:
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
