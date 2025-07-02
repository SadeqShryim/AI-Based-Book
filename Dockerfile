# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Streamlit-specific: allow web traffic
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "webui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
