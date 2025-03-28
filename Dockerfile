# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy static and templates directories for better caching
COPY static/ /app/static/
COPY templates/ /app/templates/

# Copy individual files for robots.txt and sitemap.xml
COPY robots.txt /app/
COPY sitemap.xml /app/

# Copy the rest of the application code
COPY . /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers=1", "--threads=4", "main:app"]