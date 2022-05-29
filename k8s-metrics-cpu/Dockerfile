# Pull in Python build of CPA
FROM python:3.8-slim

# Install dependencies for debugging
RUN apt-get update
RUN apt-get install jq curl -y
# Set up aliases for easy debugging
RUN echo 'alias metrics="curl -X GET http://localhost:5000/api/v1/metrics | jq ."' >> ~/.bashrc
RUN echo 'alias evaluation="curl -X POST http://localhost:5000/api/v1/evaluation | jq ."' >> ~/.bashrc

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY . /app
ENTRYPOINT ["python"]
CMD ["api.py"]

# Add config, evaluator and metric gathering Py scripts
ADD config.yaml evaluate.py metric.py GRU_Model_24 /
