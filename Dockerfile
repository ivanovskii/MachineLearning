FROM python:latest

WORKDIR /root

COPY notebooks ./notebooks

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root