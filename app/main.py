import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, RequestTimer, MetricsCollector
from app.agent import run_agent
from dotenv import load_dotenv

load_dotenv()

# Initialize configurations & singletons
settings = get_settings()
logger = get_logger("production-api")
limiter = Limiter(key_func=get_remote_address)

security_pipeline = SecurityPipeline()
response_cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
metrics_collector = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting Production API in [{settings.app_env}] environment",
        extra={"extra_data": {"primary_model": settings.primary_model}},
    )
    yield
    logger.info("Shutting down Production API")


# Create FastAPI application
app = FastAPI(
    title="Production RAG API",
    description="Production-ready RAG API with LangGraph, LangSmith observability, security pipeline, and caching.",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Security or validation rejection"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
@limiter.limit(settings.rate_limit)
@traceable(name="chat_endpoint")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Main chat endpoint handling:
    1. Response Caching
    2. Input Security Sanitization & PII Masking
    3. LangGraph Agent Execution with Memory
    4. Output Validation & PII Redaction
    5. Performance & Observability Metrics Tracking
    """
    with RequestTimer() as timer:
        # Step 1: Check cache
        cached_response = response_cache.get(body.message)
        if cached_response is not None:
            metrics_collector.record_request(
                latency_ms=timer.elapsed_ms,
                is_error=False,
                input_tokens=0,
                output_tokens=0,
                cache_hit=True,
            )
            logger.info(
                "Cache hit",
                extra={"extra_data": {"thread_id": body.thread_id, "cached": True}},
            )
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used=settings.primary_model,
                cached=True,
                processing_time_ms=round(timer.elapsed_ms, 2),
            )

        # Step 2: Input security inspection & PII masking
        is_allowed, cleaned_prompt, security_notes = security_pipeline.check_input(
            body.message
        )
        if not is_allowed:
            metrics_collector.record_request(
                latency_ms=timer.elapsed_ms,
                is_error=True,
                cache_hit=False,
            )
            logger.warning(
                "Input blocked by security pipeline",
                extra={"extra_data": {"reasons": security_notes}},
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="Security check failed",
                    detail="; ".join(security_notes),
                ).model_dump(),
            )

        try:
            # Step 3: Invoke LangGraph agent
            (
                response_text,
                model_used,
                in_tokens,
                out_tokens,
            ) = await run_agent(cleaned_prompt, thread_id=body.thread_id)

            # Step 4: Validate output
            cleaned_output, warnings = security_pipeline.check_output(response_text)
            if warnings:
                logger.warning(
                    "Output warnings detected",
                    extra={"extra_data": {"warnings": warnings}},
                )

            # Step 5: Cache verified response
            response_cache.set(body.message, cleaned_output)

            # Step 6: Record success metrics
            metrics_collector.record_request(
                latency_ms=timer.elapsed_ms,
                is_error=False,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cache_hit=False,
            )
            if security_notes:
                logger.info("Security notes",extra={"extra_data" : {
                    "note" : security_notes,
                    "thread_id" : body.thread_id,
                    "model_used" : model_used,
                    "latency_ms" : round(timer.elapsed_ms,2)
                }})
            logger.info(
                "Request processed successfully",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "latency_ms": round(timer.elapsed_ms, 2),
                        "model_used": model_used,
                    }
                },
            )

            return ChatResponse(
                response=cleaned_output,
                thread_id=body.thread_id,
                model_used=model_used,
                cached=False,
                processing_time_ms=round(timer.elapsed_ms, 2),
            )

        except Exception as exc:
            metrics_collector.record_request(
                latency_ms=timer.elapsed_ms,
                is_error=True,
                cache_hit=False,
            )
            logger.error(
                f"Error processing agent execution: {exc}",
                extra={"extra_data": {"thread_id": body.thread_id}},
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Agent execution failed",
                    detail=str(exc),
                ).model_dump(),
            )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check endpoint."""
    return HealthResponse(
        status="Healthy",
        environment=settings.app_env,
        version="1.0.0",
        checks={
            "cache": "true",
            "security": "true",
            "monitoring": "true",
        },
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    """Application metrics endpoint."""
    return MetricsResponse(**metrics_collector.summary)

@app.get("/cache/stats")
async def cache_stats():
    return response_cache.stats
