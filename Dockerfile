FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# --- START: 新增内容 ---
# 将 entrypoint.sh 脚本复制到容器中，并赋予执行权限
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
# --- END: 新增内容 ---

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports (8888 for compatibility, 7860 for Hugging Face)
EXPOSE 8888 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=7860

# Health check (use PORT environment variable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# --- START: 修改内容 ---
# 使用 entrypoint.sh 作为启动入口
ENTRYPOINT ["entrypoint.sh"]
# --- END: 修改内容 ---

# Run the application using app.py (Hugging Face compatible entry point)
CMD ["python", "app.py"]
