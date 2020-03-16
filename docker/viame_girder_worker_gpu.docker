FROM kitware/viame:girder-worker-gpu-latest

USER root

WORKDIR /home

RUN git clone https://github.com/VIAME/VIAME.git /viame \ 
	&& cd /viame/cmake \
	&& chmod +x build_server_ubuntu.sh \
	&& ./build_server_ubuntu.sh

RUN git clone https://github.com/VIAME/VIAME-Web.git /web-viame \
	&& mv /web-viame/docker/provision provision \
	&& mv /web-viame/server viame_girder 

RUN cd viame_girder && pip install --no-cache-dir -e .

USER worker

CMD girder-worker -l info
