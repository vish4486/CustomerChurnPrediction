# Use official slim Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: Install DVC for version control of data
RUN pip install dvc

# Optional: Run DVC pull if remote is configured (commented by default)
# RUN dvc pull

# Expose default Streamlit port
EXPOSE 8501

# Default command to launch Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
