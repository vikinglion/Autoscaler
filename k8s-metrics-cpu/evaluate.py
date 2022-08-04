import json
import sys
import math
import pandas as pd
from numpy import array
from tensorflow.python.keras.saving.save import load_model

target_average_utilization = 0.5

def main():
    model_path = "./GRU_Model_24"
    predictModel = load_model(model_path)
    # Parse JSON into a dict
    spec = json.loads(sys.stdin.read())
    evaluate(spec,predictModel)

def readData():
    inputList = []
    inputLength = 0
    try:
        df = pd.read_csv('./data.csv', header=None)
        inputList = df[0].tolist()
        inputLength = len(inputList)
    except:
        sys.stderr.write("read Data failed!!!")
        exit(1)
    return inputList, inputLength

def ScaleLogic(currentReplicas,currentMetric):
    return math.ceil(currentReplicas * (currentMetric / target_average_utilization))

def PredictLogic(currentReplicas, inputSize, predictModel):
    targetReplica = 0
    collectedLoad, loadLength = readData()

    if predictModel and loadLength >= inputSize:
        inputLength = inputSize
        sequence = collectedLoad[-inputLength:] or []
        X_val = array(sequence)
        X_val = X_val.reshape((1, inputLength, 1))
        predictMetric = predictModel.predict(X_val)
        targetReplica = ScaleLogic(currentReplicas, float(predictMetric))

    return targetReplica

def evaluate(spec, predictModel):
    inputSize = 24
    
    # Only expect 1 metric provided
    if len(spec["metrics"]) != 1:
        sys.stderr.write("Expected 1 metric")
        exit(1)

    # Get the metric value, there should only be 1
    metric_value = json.loads(spec["metrics"][0]["value"])

    # Get the current replicas from the metric
    current_replicas = metric_value["current_replicas"]
    # Get the average utilization and total utilization from the metric
    average_utilization = metric_value["average_utilization"]

    # Calculate target replicas by forecasting model
    predict_replicas = PredictLogic(current_replicas, inputSize, predictModel)
    target_replicas = predict_replicas
    
    # HPA logic
    if target_replicas == 0:
        target_replicas = ScaleLogic(current_replicas,average_utilization)

    # direct scaling
    # target_replicas = current_replicas
    # if average_utilization > target_average_utilization:
    #     target_replicas += 1
    # else:
    #     target_replicas -= 1
    
    # Build JSON dict with targetReplicas
    evaluation = {}
    evaluation["targetReplicas"] = target_replicas

    # Output JSON to stdout
    sys.stdout.write(json.dumps(evaluation))

if __name__ == "__main__":
    main()
