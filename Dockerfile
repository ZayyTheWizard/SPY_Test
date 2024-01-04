FROM python:3.11.6-slim-bookworm

WORKDIR /app

COPY requirments.txt .

RUN pip install --no-cache-dir -r requirments.txt

COPY . .

CMD ["python", "./main.py"]