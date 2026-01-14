# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet omw-1.4

# Copy project code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Fix src imports
ENV PYTHONPATH=/app/src

# Run Streamlit app
CMD ["streamlit", "run", "src/ML_Model_Copilot/frontend/streamlit_app.py", "--server.address=0.0.0.0"]
