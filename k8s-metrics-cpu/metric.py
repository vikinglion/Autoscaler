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
import pandas as pd

def collectData(data, inputSize):
    data = pd.DataFrame([data])
    isNewData = False
    try:
        df = pd.read_csv('./data.csv', header=None)
        inputList = df[0].tolist()
        print(inputList)
        print("Data size is {size}".format(size = df.shape[0]))

    except:
        data.to_csv('./data.csv', mode="w", index=False, header=False)
        df = pd.read_csv('./data.csv', header=None)
        isNewData = True

    if not isNewData:
        df.dropna(inplace = True)
        if df.shape[0] >= inputSize:
            df = df.drop([0])

        df = df.append(data)
        df.to_csv('./data.csv', mode="w", index=False, header=False)

def main():
    # Parse JSON into a dict
    spec = json.loads(sys.stdin.read())
    metric(spec)

def metric(spec):
    # Get the Kubernetes metrics value, there is only 1 expected, so it should be the first one
    cpu_metrics = spec["kubernetesMetrics"][0]
    # Pull out the current replicas
    current_replicas = cpu_metrics["current_replicas"]
    # Get the resource metric info
    resource = cpu_metrics["resource"]
    # Get the list of pod metrics
    pod_metrics_info = resource["pod_metrics_info"]
    # Total up all of the pod values
    total_utilization = 0
    for _, pod_info in pod_metrics_info.items():
        total_utilization += pod_info["Value"]
    # Calculate the average utilization
    average_utilization = total_utilization / current_replicas

    collectData(average_utilization, 24)

    # Generate some JSON to pass to the evaluator
    sys.stdout.write(json.dumps(
        {
            "current_replicas": current_replicas,
            "average_utilization": average_utilization,
            "total_utilization" : total_utilization
        }
    ))

if __name__ == "__main__":
    main()
