FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy source code
COPY src/ ./src/

EXPOSE 7860

CMD ["python", "-m", "src.webapp.app"]
