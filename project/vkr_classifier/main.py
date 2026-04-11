from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
import uvicorn

from vkr_classifier import __version__
from vkr_classifier.api import build_api_router
from vkr_classifier.config import Settings, get_settings
from vkr_classifier.service import ClassifierService
from vkr_classifier.ui import build_ui


def create_application(
    settings: Settings | None = None,
    include_ui: bool = True,
) -> FastAPI:
    effective_settings = settings or get_settings()
    service = ClassifierService(effective_settings)
    service.ensure_ready()

    app = FastAPI(
        title="VKR Classifier",
        version=__version__,
        description="Классификация текстов и изображений с low-code интерфейсом.",
    )
    app.state.classifier_service = service
    app.include_router(build_api_router(service))

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        target = "/ui" if include_ui else "/docs"
        return RedirectResponse(url=target)

    if include_ui:
        demo = build_ui(service)
        app = gr.mount_gradio_app(app, demo, path="/ui")

    return app


def main() -> None:
    settings = get_settings()
    uvicorn.run(create_application(settings=settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()

