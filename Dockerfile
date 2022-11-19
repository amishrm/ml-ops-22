FROM python:3.8.1
LABEL maintainer="Amit Sharma <sharma.95@iitj.ac.in>"
EXPOSE 8000
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
COPY . /app
WORKDIR /app
#CMD ["python3", "./mlops/plot_graphs.py"]
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
CMD ["python3", "app.py"]