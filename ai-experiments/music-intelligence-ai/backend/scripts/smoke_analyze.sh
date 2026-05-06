#!/usr/bin/env bash
# HTTP smoke: requires uvicorn on 127.0.0.1:8000. Creates a 1s silent WAV and POSTs to /analyze.
set -euo pipefail
TMP="$(mktemp /tmp/safari-analyze-XXXX.wav)"
python3 -c "
import io, wave, sys
buf = io.BytesIO()
with wave.open(buf, 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(22050)
    w.writeframes(b'\\x00\\x00' * 22050)
open(sys.argv[1], 'wb').write(buf.getvalue())
" "$TMP"
curl -sS -X POST -F "file=@${TMP}" http://127.0.0.1:8000/analyze | python3 -m json.tool
rm -f "$TMP"
