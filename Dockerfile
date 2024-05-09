FROM tiangolo/uvicorn-gunicorn:python3.9

RUN mkdir /fastapi

RUN git clone https://github.com/deepanshu-yadav/spacesense_mlops_challenge.git /tmp/repo

COPY /tmp/repo /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt 

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
