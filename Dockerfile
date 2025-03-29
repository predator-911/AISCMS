FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages with --no-cache-dir and force-reinstall numpy
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --force-reinstall numpy==1.24.4 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
