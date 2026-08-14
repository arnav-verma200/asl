FROM python:3.9-slim

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model directories
COPY app/ app/
COPY models/ models/
COPY data/ data/

# Set Flask environment variable
ENV FLASK_APP=app/app.py

# Expose port
EXPOSE 5000

# Start server using Flask CLI to bind to 0.0.0.0
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
