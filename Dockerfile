FROM python:3.12-slim
WORKDIR /app
COPY opbench ./opbench
COPY web ./web
COPY templates ./templates
RUN useradd --uid 10001 --create-home opbench && mkdir /data && chown opbench:opbench /data
USER opbench
EXPOSE 30000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:30000/api/health',timeout=3)"
CMD ["python", "-m", "opbench.server", "--host", "0.0.0.0", "--db", "/data/opbench.sqlite3"]
