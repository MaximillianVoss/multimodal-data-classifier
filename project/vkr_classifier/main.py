from __future__ import annotations

import socket
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
        app = gr.mount_gradio_app(
            app,
            demo,
            path="/ui",
            theme=gr.themes.Soft(),
        )

    return app


def _can_bind(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _resolve_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    if _can_bind(host, preferred_port):
        return preferred_port

    for candidate in range(preferred_port + 1, preferred_port + attempts + 1):
        if _can_bind(host, candidate):
            print(f"Порт {preferred_port} занят. Приложение будет запущено на http://{host}:{candidate}")
            return candidate

    raise RuntimeError(
        f"Не удалось найти свободный порт в диапазоне {preferred_port}-{preferred_port + attempts}."
    )


def main() -> None:
    settings = get_settings()
    port = _resolve_port(settings.host, settings.port)
    uvicorn.run(create_application(settings=settings), host=settings.host, port=port)


if __name__ == "__main__":
    main()
