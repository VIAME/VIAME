FROM girder/girder:latest

WORKDIR /home

RUN pip install --no-cache-dir \
        girder-jobs \
        girder-worker \
    && girder build

# modify this based on where you are running docker-compose from
COPY docker/provision provision

COPY server viame_girder

RUN cd viame_girder && pip install --no-cache-dir .

# Build the client and serve it via girder
COPY client viame_client

RUN cd viame_client && npm install && npm run build && \
    mkdir /usr/share/girder/static/viame && \
    cp -r dist/* /usr/share/girder/static/viame/

ENTRYPOINT ["/home/provision/girder_entrypoint.sh"]
