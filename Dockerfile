FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "model2vec>=0.6.0" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "numpy>=1.26"

COPY server/ /app/server/
COPY kor-minish-bge-m3-ko/ /app/kor-minish-bge-m3-ko/

ENV KOR_MINISH_MODEL=/app/kor-minish-bge-m3-ko
ENV KOR_MINISH_BATCH=256

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
