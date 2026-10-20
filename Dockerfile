FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch: free hosting has no GPU, and the CUDA build is ~4x bigger.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies first, so code changes don't invalidate this cached layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY app ./app
COPY .streamlit ./.streamlit
RUN pip install --no-deps -e .

# Don't run as root (uid 1000 is also what Hugging Face Spaces expects).
RUN useradd -m -u 1000 user
USER user

# Models are not baked into the image: mount them at /app/models.
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
