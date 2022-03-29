FROM python:3.8-buster
COPY . /home
WORKDIR /home
RUN pip install -r /home/requirements.txt --ignore-installed

ENTRYPOINT ["uvicorn api:api"]