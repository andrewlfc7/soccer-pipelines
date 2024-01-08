FROM python:3.11

LABEL authors="andrew"

# Set the working directory in the container
WORKDIR /app

# Copy everything from the context into the /app directory
COPY . .


# Install dependencies
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install --no-cache-dir -r requirements.txt


# Set the environment variable
ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 80

# Command to run the application

CMD ["sh", "-c", "python3 main.py"]


