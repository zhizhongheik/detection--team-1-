FROM python:3.11-slim

 RUN apt-get update && apt-get install --upgrade pip -y  libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -no-cache-dir -r requirements.txt

# copy all the project files into the container
COPY . .

# exposing

EXPOSE 8000

# start 

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]

