from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from vkr_classifier.schemas import (
    BatchHistoryEntry,
    BatchRunResponse,
    HealthResponse,
    HistoryEntry,
    ModelInfo,
    PredictionResponse,
    TextClassificationRequest,
)
from vkr_classifier.service import ClassifierService


def build_api_router(service: ClassifierService) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["classifier"])

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", models_ready=service.is_ready)

    @router.get("/models", response_model=list[ModelInfo])
    async def models() -> list[ModelInfo]:
        return [ModelInfo(**row) for row in service.get_models()]

    @router.get("/history", response_model=list[HistoryEntry])
    async def history(limit: int = 20) -> list[HistoryEntry]:
        return [HistoryEntry(**row) for row in service.get_history(limit=limit)]

    @router.get("/batch/history", response_model=list[BatchHistoryEntry])
    async def batch_history(limit: int = 10) -> list[BatchHistoryEntry]:
        return [BatchHistoryEntry(**row) for row in service.get_batch_history(limit=limit)]

    @router.post("/text/classify", response_model=PredictionResponse)
    async def classify_text(payload: TextClassificationRequest) -> PredictionResponse:
        try:
            return PredictionResponse(**service.classify_text(payload.text))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/image/classify", response_model=PredictionResponse)
    async def classify_image(file: UploadFile = File(...)) -> PredictionResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Нужно загрузить файл изображения.")
        try:
            data = await file.read()
            return PredictionResponse(**service.classify_image_bytes(data))
        except Exception as error:  # pragma: no cover - внешние ошибки PIL
            raise HTTPException(status_code=400, detail=f"Не удалось обработать изображение: {error}") from error

    @router.post("/batch/classify-archive", response_model=BatchRunResponse)
    async def classify_archive(file: UploadFile = File(...)) -> BatchRunResponse:
        if not file.filename or not file.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Нужно загрузить ZIP-архив документов.")

        from tempfile import NamedTemporaryFile
        from pathlib import Path

        temp_path: Path | None = None
        try:
            payload = await file.read()
            with NamedTemporaryFile(delete=False, suffix=".zip") as temporary_file:
                temporary_file.write(payload)
                temp_path = Path(temporary_file.name)
            result = service.classify_archive(temp_path, source_name=file.filename)
            return BatchRunResponse(**result)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:  # pragma: no cover - файловые ошибки среды
            raise HTTPException(status_code=400, detail=f"Не удалось обработать архив: {error}") from error
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    return router
