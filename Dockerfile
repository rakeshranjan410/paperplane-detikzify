FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Pillow and others if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY detikzify/ detikzify/
COPY start_server.sh .

# Install dependencies
# We use instructions from pyproject.toml
# Set version to avoid setuptools-scm error in Docker (no .git)
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DETIKZIFY="0.3.0"
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Run
CMD ["./start_server.sh"]
