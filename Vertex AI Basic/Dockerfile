FROM pytorch/torchserve:latest-cpu

COPY ./handler.py /home/model-server/
COPY ./model.pt /home/model-server/
COPY ./index_to_name.json /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

EXPOSE 7080
EXPOSE 7081

RUN torch-model-archiver -f \
  --model-name=model \
  --version=1.0 \
  --serialized-file=/home/model-server/model.pt \
  --handler=/home/model-server/handler.py \
  --export-path=/home/model-server/model-store

  CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "model=model.mar", \
     "--model-store", \
     "/home/model-server/model-store"]