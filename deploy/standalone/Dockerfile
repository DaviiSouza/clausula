# All-in-one image (Cloud Run, HF Spaces): API (internal, port 8000) + Streamlit (public, $PORT, default 7860).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLAUSULA_API_URL=http://127.0.0.1:8000

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user . .
RUN pip install --no-deps -e .

USER user
# Download the model at build time so the Space starts faster.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('DaviiiiS/clausula-legal-bert')"

EXPOSE 7860
CMD ["sh", "start.sh"]
