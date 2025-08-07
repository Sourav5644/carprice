FROM python:3.10-slim

WORKDIR /app

# Copy app code
COPY flask_app/ /app/

# Create the 'processed' directory in the container
RUN mkdir -p /app/processed

# Copy the preprocessor into the container
COPY data/processed/preprocessor.pkl /app/data/processed/preprocessor.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Local
CMD ["python", "app.py"]

# Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
