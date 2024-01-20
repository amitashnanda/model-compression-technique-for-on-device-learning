FROM python:3.9-slim

WORKDIR D:/"Tiny ML"/Gatech_Code/app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "./Training/select_model.py"]