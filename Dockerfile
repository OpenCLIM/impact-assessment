FROM continuumio/miniconda3:4.8.2

RUN mkdir src

WORKDIR src

COPY environment.yml .
RUN conda env update -f environment.yml -n base

COPY script.py .

CMD [ "python", "script.py"]
