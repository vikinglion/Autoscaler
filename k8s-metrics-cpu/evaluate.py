# Copyright 2021 The Custom Pod Autoscaler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
import math
import pandas as pd
from numpy import array
from tensorflow.python.keras.saving.save import load_model


# JSON piped into this script example:
# {
#   "resource": "php-apache",
#   "runType": "api",
#   "metrics": [
#     {
#       "resource": "php-apache",
#       "value": "{\"current_replicas\": 3, \"average_utilization\": 60}"
#     }
#   ]
# }

target_average_utilization = 25

def main():
    model_name = "GRU_Model_24"
    pridectModel = load_model(model_name)
    # Parse JSON into a dict
    spec = json.loads(sys.stdin.read())
    evaluate(spec,pridectModel)

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

def PredictLogic(currentReplicas, inputSize, pridectModel):
    targetReplica = 0
    collectedLoad, loadLength = readData()

    if loadLength >= inputSize:
        inputLength = inputSize
        sequence = collectedLoad[-inputLength:] or []
        X_val = array(sequence)
        X_val = X_val.reshape((1, inputLength, 1))
        predictMetric = pridectModel.predict(X_val)
        targetReplica = ScaleLogic(currentReplicas, float(predictMetric))

    return targetReplica

def evaluate(spec, pridectModel):
    inputSize = 24
    
    # Only expect 1 metric provided
    if len(spec["metrics"]) != 1:
        sys.stderr.write("Expected 1 metric")
        exit(1)

    # Get the metric value, there should only be 1
    metric_value = json.loads(spec["metrics"][0]["value"])

    # Get the current replicas from the metric
    current_replicas = metric_value["current_replicas"]
    # Get the average utilization from the metric
    average_utilization = metric_value["average_utilization"]

    # Calculate target replicas, increase by 1 if utilization is above target, decrease by 1 if utilization is below
    # target

    predict_replicas = PredictLogic(current_replicas, inputSize, pridectModel)
    target_replicas = predict_replicas

    if target_replicas == 0:
        # direct scaling
        target_replicas = current_replicas
        if average_utilization > target_average_utilization:
            target_replicas += 1
        else:
            target_replicas -= 1

    # Build JSON dict with targetReplicas
    evaluation = {}
    evaluation["targetReplicas"] = target_replicas

    # Output JSON to stdout
    sys.stdout.write(json.dumps(evaluation))

if __name__ == "__main__":
    main()
