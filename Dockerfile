FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required by pypdf or chromadb
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and data
COPY . .

# Expose Gradio's default port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
