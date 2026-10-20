#!/bin/sh
# Start the API in the background and wait until the model is loaded before opening the app.
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
python -c "
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/health'); break
    except Exception:
        time.sleep(1)
"
exec streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port "${PORT:-7860}" \
    --server.enableXsrfProtection false --server.enableCORS false
