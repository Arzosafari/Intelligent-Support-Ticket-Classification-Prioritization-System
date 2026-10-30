"""Local assistive API: no raw text logging, no persistence, no business actions."""

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from support_ticket.api.schemas import BatchRequest, BatchResponse, Prediction, Ticket
from support_ticket.models.predict import Predictor

logger = logging.getLogger("support_ticket.api")


class RequestGuard:
    """Bound streamed bodies too; Content-Length alone is not a safe size limit."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        start = time.perf_counter()
        status = 500

        async def wrapped_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message["headers"]) + [
                    (b"x-request-id", request_id.encode())
                ]
            await send(message)

        # Buffer only up to configured bound, then replay to the framework.
        chunks, size = [], 0
        try:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > self.max_bytes:
                    response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                    await response(scope, receive, wrapped_send)
                    return
                chunks.append(message)
                if not message.get("more_body", False):
                    break

            async def replay() -> Message:
                return chunks.pop(0) if chunks else await receive()

            await self.app(scope, replay, wrapped_send)
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": scope["method"],
                        "status": status,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                    }
                )
            )


def create_app(artifact_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        try:
            app.state.predictor = Predictor(
                artifact_path or os.getenv("MODEL_DIR", "models/current")
            )
        except Exception:
            # Do not expose paths, artifact content or deserialization errors.
            logger.error(json.dumps({"event": "artifact_unavailable"}))
        yield
        app.state.predictor = None

    app = FastAPI(title="Intelligent Support Ticket System", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestGuard, max_bytes=int(os.getenv("MAX_BODY_BYTES", "1100000")))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI defaults include the invalid input; redact it to protect customer content.
        errors = [
            {"loc": list(e["loc"]), "type": e["type"], "msg": "Invalid request value"}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(json.dumps({"event": "internal_error"}))
        return JSONResponse(status_code=500, content={"detail": "Internal service error"})

    def predictor() -> Predictor:
        instance = getattr(app.state, "predictor", None)
        if instance is None:
            raise HTTPException(503, "Model unavailable; contact operator")
        return instance

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        predictor()
        return {"status": "ready"}

    @app.get("/model-info")
    def info() -> dict:
        return predictor().safe_info()

    async def infer(tickets: list[Ticket]) -> list[dict]:
        instance = predictor()
        try:
            predictions = await run_in_threadpool(instance.predict, [t.text for t in tickets])
        except ValueError:
            raise HTTPException(422, "Input violates model constraints") from None
        for ticket, result in zip(tickets, predictions, strict=True):
            result["ticket_id"] = ticket.ticket_id
        logger.info(
            json.dumps(
                {
                    "event": "prediction_batch",
                    "count": len(predictions),
                    "model_version": instance.metadata["model_version"],
                    "review_count": sum(p["category"]["review_required"] for p in predictions),
                    "categories": [p["category"]["label"] for p in predictions],
                }
            )
        )
        return predictions

    @app.post("/predict", response_model=Prediction)
    async def predict(ticket: Ticket) -> dict:
        return (await infer([ticket]))[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    async def batch(request: BatchRequest) -> dict:
        return {"predictions": await infer(request.tickets)}

    return app


app = create_app()
