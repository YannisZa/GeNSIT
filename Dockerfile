FROM python:3.9.7-slim

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get -y install gcc \
&& rm -rf /var/lib/apt/lists/* \
&& /usr/local/bin/python -m pip install --upgrade pip

COPY . .

RUN pip3 install --no-cache-dir --compile -e .

RUN gcc -fPIC -shared -o ./ticodm/sim_models/production_constrained/spatial_interaction.so ./ticodm/sim_models/production_constrained/spatial_interaction.c -O3
RUN gcc -fPIC -shared -o ./ticodm/helper_c_functions/helper_functions.so ./ticodm/helper_c_functions/helper_functions.c -O3

ENTRYPOINT ["ticodm"]
