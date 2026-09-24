FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY support_assistant ./support_assistant

WORKDIR /app/support_assistant
EXPOSE 7860

CMD ["python", "main.py"]
