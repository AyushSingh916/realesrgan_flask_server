# Use Python 3.10.11 slim image
FROM python:3.10.11
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# Set the working directory in the container
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt
# Install system dependencies and Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt 

COPY . /app
# Expose the port that Flask will run on

RUN update_degradations.py

EXPOSE 5000
# Command to run the Flask app
CMD ["python", "app.py"]