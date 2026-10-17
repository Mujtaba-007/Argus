# Dockerfile for the Argus backend service.
# Build:  docker build -t argus-backend .
# Run:    docker run --env-file .env -p 8000:8000 argus-backend

FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /backend

# Install dependencies first (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY app/ ./app

# Expose the port uvicorn listens on
EXPOSE 8080

# Health check — Docker will mark the container unhealthy if /health stops responding.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
