from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.stream import router as stream_router

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(stream_router, prefix="")


@app.get("/health")
def health_check() -> dict:
	return {"status": "ok"}
