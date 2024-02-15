FROM python:3.9.7-slim

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get -y install gcc \
&& rm -rf /var/lib/apt/lists/* \
&& /usr/local/bin/python -m pip3 install --upgrade pip3

COPY . .

RUN pip3 install --no-cache-dir --compile -e .

ENTRYPOINT ["gensit"]
