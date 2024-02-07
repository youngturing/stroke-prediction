FROM tensorflow/tensorflow

LABEL authors="Mikołaj Daraż"

WORKDIR /code

COPY ../requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt &&  \
    rm /code/requirements.txt

COPY ../model /code/model
COPY ../app /code/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
