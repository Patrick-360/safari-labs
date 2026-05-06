from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.stream import router as stream_router
from app.cors_config import cors_allow_origins

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=cors_allow_origins(),
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(stream_router, prefix="")
app.include_router(analyze_router, prefix="")


@app.get("/health")
def health_check() -> dict:
	return {"status": "ok"}
