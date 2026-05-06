"""CORS allow-origins from environment (comma-separated)."""

from __future__ import annotations

import os


def cors_allow_origins() -> list[str]:
	"""
	Parse CORS_ORIGINS, e.g. "http://localhost:3000" or
	"http://localhost:3000,https://app.example.com".
	Empty entries are ignored.
	"""
	raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
	origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
	return origins if origins else ["http://localhost:3000"]
