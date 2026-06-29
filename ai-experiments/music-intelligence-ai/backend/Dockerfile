FROM python:3.11-slim

# ffmpeg: pre-trims uploaded audio to 60 s before librosa sees it, keeping
# peak RAM under ~10 MB for audio regardless of upload size.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
