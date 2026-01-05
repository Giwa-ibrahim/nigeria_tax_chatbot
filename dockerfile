# Use Python 3.11 slim image for smaller size
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user first
RUN useradd -m -u 1000 appuser

# Create necessary directories
RUN mkdir -p chroma_db dataset/processed_data


# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for FastAPI
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
