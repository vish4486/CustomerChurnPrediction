# Use an official Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: Install DVC if used
RUN pip install dvc

# Optional: Run DVC pull if remote configured
# RUN dvc pull

# Default command (you can override it later)
CMD ["python", "app/main.py"]

