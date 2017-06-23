FROM ubuntu:16.04

MAINTAINER codeworm96 <codeworm96@outlook.com>

WORKDIR /root

RUN apt-get update && \
    apt-get install -y python3-pip

COPY . .

RUN pip3 install jieba flask requests tensorflow keras pandas h5py gensim

ENV LANG C.UTF-8

CMD [ "sh", "-c", "bash"]
