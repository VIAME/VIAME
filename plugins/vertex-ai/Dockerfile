# VIAME Vertex AI Custom Container
#
# Build:
#   docker build -t us-docker.pkg.dev/PROJECT/REPO/viame-vertex-ai:latest .
#
# Test locally:
#   docker run -p 8080:8080 -e AIP_HTTP_PORT=8080 viame-vertex-ai:latest

FROM viame/viame-gpu:latest

RUN pip install --no-cache-dir \
  flask>=2.3 \
  gunicorn>=21.2 \
  google-cloud-storage>=2.10 \
  google-cloud-aiplatform>=1.38

COPY plugins/vertex-ai/ /opt/noaa/viame/plugins/vertex-ai/

WORKDIR /workspace

ENV VIAME_INSTALL_DIR=/opt/noaa/viame
ENV VIAME_WORK_DIR=/workspace
ENV AIP_HTTP_PORT=8080

EXPOSE 8080

ENTRYPOINT ["bash", "-c", \
  "source /opt/noaa/viame/setup_viame.sh && \
   cd /opt/noaa/viame/plugins/vertex-ai && \
   gunicorn --bind 0.0.0.0:${AIP_HTTP_PORT} \
            --timeout 600 \
            --workers 1 \
            --threads 4 \
            app:app"]
