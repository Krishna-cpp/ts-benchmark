# Reproduce the full benchmark (classical models) in a clean environment:
#   docker build -t tsbench .
#   docker run --rm -v "$(pwd)/results:/app/results" tsbench
# Add Granite TTM with:  docker build --build-arg EXTRAS=foundation -t tsbench .
FROM python:3.11-slim

ARG EXTRAS=dev
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

COPY pyproject.toml README.md ./
COPY tsbench ./tsbench
RUN if [ "$EXTRAS" = "foundation" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && pip install ".[$EXTRAS]"

CMD ["python", "-m", "tsbench"]
