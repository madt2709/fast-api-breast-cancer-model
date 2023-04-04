# Start with python base image
FROM python:3.10

# Set working directory to /code
WORKDIR /code

# Copy the requirements to /code
ADD requirements.txt .

# --no-cache-dir option tells pip to not save the downloaded packages locally.
# --upgrade means we upgrade existing packages that might have been saved in the Docker cache.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy main and model parameters
COPY . .

# Listen to port 8000 at runtime
EXPOSE 8000

# Add uvicorn as an entrypoint.
ENTRYPOINT ["uvicorn"]

# Start the app
CMD ["main:app", "--port", "80"]