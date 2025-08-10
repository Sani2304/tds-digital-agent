# Use the official Python 3.11 slim image (small size version of Python)
FROM python:3.11-slim

# Stop Python from creating .pyc files and make output show immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the folder where our app will run inside the container
WORKDIR /app

# Install system packages needed for pandas, lxml, and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \        # for compiling some Python packages
    libxml2-dev \            # for parsing XML/HTML
    libxslt-dev \            # for lxml support
    && rm -rf /var/lib/apt/lists/*  # clean up to make image smaller

# Copy the requirements file first so Docker can cache installs
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the app's Python code into the container
COPY app/ app/

# Let Docker know our app will run on port 8000
EXPOSE 8000

# Command to start the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
