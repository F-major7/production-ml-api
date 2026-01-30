# Multi-stage build for Production ML API
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime - Create final image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN useradd -m -u 1000 apiuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/apiuser/.local

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY db/ ./db/
COPY cache/ ./cache/
COPY monitoring/ ./monitoring/
COPY alembic.ini .

# Change ownership of all app files
RUN chown -R apiuser:apiuser /app /home/apiuser/.local

# Switch to non-root user
USER apiuser

# Make sure scripts in .local are usable
ENV PATH=/home/apiuser/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check disabled - triggers NumPy/PyTorch segfault in containerized environment
# Prometheus monitoring provides service health instead
# HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Run application with asyncio loop (NOT uvloop - causes PyTorch segfaults in Docker)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]

