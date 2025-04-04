FROM ubuntu:22.04

# System dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip && \
    apt-get clean

# Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Application code
COPY . /app
WORKDIR /app

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
