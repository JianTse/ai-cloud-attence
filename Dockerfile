FROM jantes/mxnet-python-opencv

WORKDIR /usr/src/app

COPY . .

CMD [ "python", "./run.py" ]