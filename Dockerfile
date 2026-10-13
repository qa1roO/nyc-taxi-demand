FROM python:3.12-slim-bookworm

# Spark runs on the JVM, so the image needs Java; LightGBM needs the OpenMP runtime (libgomp1)
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# PostgreSQL JDBC driver: lets Spark write tables straight into PostgreSQL
ADD https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar /opt/jars/
ENV POSTGRES_JDBC_JAR=/opt/jars/postgresql-42.7.4.jar

CMD ["bash"]
