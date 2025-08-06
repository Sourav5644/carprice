# # Use an official Pyhton 3.10 image from docker hub
# FROM python:3.10-slim-buster

# #Set the working directory
# WORKDIR /app

# # copy application code
# COPY . /app

# #Install the dependencies
# RUN pip install -r requirements.txt

# # Expose the port FastApi will run on
# EXPOSE 5000

# # Command to run the FastApi app
# CMD ["python3", "app.py"]
# #CMD["uvicorn","app:app","--host", "0.0.0.0","--port","8080"]


