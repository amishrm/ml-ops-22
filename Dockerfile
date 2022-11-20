FROM python:3.8.1
LABEL maintainer="Amit Sharma <sharma.95@iitj.ac.in>"

EXPOSE 5000

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . /app
WORKDIR /app
#ENTRYPOINT ["python3", "Q4.py","-c","(clf_name) ","-r","(random_state)"]

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]