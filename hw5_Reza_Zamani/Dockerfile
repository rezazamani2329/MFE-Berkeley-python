# Dockerfile
# ----------
# Defines how to build a Docker image for the FastAPI ML prediction service.
# Each instruction below adds a new layer to the image.
#
# Build the image with:   docker build -t fastapi-app .
# Run a container with:   docker run -d -p 8000:8000 fastapi-app

# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------
# Start from the official Python 3.12 image in its "slim" variant.
# "slim" strips out many OS-level tools that aren't needed at runtime
# (compilers, manpages, etc.), keeping the final image significantly smaller
# than the full python:3.12 image while still including everything Python needs.
FROM python:3.12-slim

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
# All subsequent instructions (COPY, RUN, CMD) will be executed relative to
# this path inside the container. If /app doesn't exist Docker creates it.
# Using a dedicated directory keeps our files separate from OS-level paths.
WORKDIR /app

# ---------------------------------------------------------------------------
# Copy project files into the image
# ---------------------------------------------------------------------------
# Files are copied in a deliberate order to take advantage of Docker's
# layer caching. Docker re-uses cached layers for steps whose inputs haven't
# changed, so frequently changing files are copied last.

# requirements.txt changes rarely — copy it first so the pip install step
# below is only re-run when dependencies actually change.
COPY requirements.txt .

# app.py is the FastAPI application code.
COPY app.py .

# model.joblib is the serialised ML model loaded by app.py at startup.
COPY model.joblib .

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
# Runs pip inside the image to install everything listed in requirements.txt.
#   --no-cache-dir  prevents pip from storing the download cache on disk,
#                   which reduces the final image size since that cache would
#                   serve no purpose inside a container.
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Start-up command
# ---------------------------------------------------------------------------
# CMD defines the default command that runs when a container starts.
# Using the JSON array ("exec") form is preferred over the shell form because
# it runs uvicorn directly (no shell wrapper), so signals like SIGTERM are
# delivered straight to the process — enabling graceful shutdown.
#
#   uvicorn      — ASGI server that serves the FastAPI application
#   app:app      — "from app.py import app" (module:variable)
#   --host 0.0.0.0  — listen on all network interfaces inside the container,
#                     not just localhost, so the port is reachable from outside
#   --port 8000  — the port uvicorn listens on (mapped to the host via -p in `docker run`)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
