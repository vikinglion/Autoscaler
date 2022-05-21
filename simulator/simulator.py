from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import random
import math
import time
import pickle
import numpy as np
from numpy import array
from matplotlib import pyplot as plt
from apscheduler.schedulers.background import BackgroundScheduler
from tensorflow.python.keras.saving.save import load_model

class Simulator():
    def __init__(self, rawData, model, inputSize):
        self.rawData = rawData
        self.dataLength = len(rawData)
        self.scheduler = BackgroundScheduler(timezone='Asia/Shanghai')
        self.pridectModel = model
        self.inputSize = inputSize

        self.loadInd = 0
        self.stopFlag = False
        self.collectedLoad = []

        self.hpaReplica = 0
        self.cpaReplica = 0
        self.predictMetric = 0

        # stablization window
        self.hpaTimer = 0
        self.cpaTimer = 0

        # deployment config
        self.replicas = 1
        self.targetMetric = 0
        self.SW = 0

    def TimedTasks(self, minute):
        self.scheduler.add_job(self.GenerateLoad, 'interval', minutes = minute)
        self.scheduler.start()

    def GenerateLoad(self):
        index = self.loadInd
        load = self.rawData[index]
        self.loadInd += 1
        if self.loadInd == len(self.rawData):
            self.stopFlag = True
            print("Stop generating loads!!")
        else:
            print("Successfully generate one load: ", load)
            print("Predict Load: ", self.predictMetric)

            self.MetricCollector(load)

    def MetricCollector(self, singleLoad):
        collectData(singleLoad, self.hpaReplica, self.cpaReplica, self.predictMetric)
        self.collectedLoad.append(singleLoad)
        if len(self.collectedLoad) >= 200:
            self.stopFlag = True
        # current metric
        print("Current Time Point: ", self.loadInd)
        print("HPA copies: ", self.hpaReplica)
        print("CPA copies: ", self.cpaReplica)
        self.Evaluator(singleLoad)

    def SetResources(self, replica = 1, targetMetric = 0.5, stabilizationWindow = 0):
        # defalut replias
        self.replicas = replica
        # target metric
        self.targetMetric = targetMetric
        self.SW = stabilizationWindow

    def Evaluator(self, currentMetric):
        threadPool.submit(self.Simu_HPA, currentMetric)
        threadPool.submit(self.Simu_CPA, currentMetric)

    def ScaleLogic(self, currentMetric):
        return math.ceil(self.replicas * (currentMetric / self.targetMetric))

    def Simu_HPA(self, currentMetric):
        targetReplica = self.ScaleLogic(currentMetric)
        self.PerformScaling()
        self.hpaReplica = self.StabilizationWindow(self.hpaReplica, targetReplica, 1)

    def Simu_CPA(self, currentMetric):
        inputLength = self.inputSize
        if len(self.collectedLoad) >= self.inputSize:
            sequence = self.collectedLoad[-inputLength:] or []
            X_val = array(sequence)
            X_val = X_val.reshape((1, inputLength, 1))
            predictMetric = self.pridectModel.predict(X_val)
        else:
            predictMetric = currentMetric
        self.predictMetric = predictMetric
        targetReplica = self.ScaleLogic(float(predictMetric))
        self.PerformScaling()
        self.cpaReplica = self.StabilizationWindow(self.cpaReplica, targetReplica, 2)

    def PerformScaling(self):
        time.sleep(5)

    def TickTimer(self):
        self.hpaTimer = 0 if self.hpaTimer <= 0 else self.hpaTimer - 1
        # print("Tick HPA Timer: %d" %(self.hpaTimer))
        self.cpaTimer = 0 if self.cpaTimer <= 0 else self.cpaTimer - 1
        # print("Tick CPA Timer: %d" %(self.cpaTimer))
        time.sleep(1)

    def StabilizationWindow(self, currentReplica, targetReplica, timerType):
        if timerType == 1:
            if currentReplica > targetReplica and self.hpaTimer > 0:
                targetReplica = currentReplica
            elif currentReplica < targetReplica:
                self.hpaTimer = self.SW
                print("HPA Timer Reseting!!!")
        elif timerType == 2:
            if currentReplica > targetReplica and self.cpaTimer > 0:
                targetReplica = currentReplica
            elif currentReplica < targetReplica:
                self.cpaTimer = self.SW
                print("CPA Timer Reseting!!!")
        return targetReplica

def readData():
    inputList = []
    try:
        df = pd.read_csv('./data.csv', header=None)
        inputList = df[0].tolist()
    except:
        print("read Data failed!!!")
    return inputList

def collectData(load, hpaReplica, cpaReplica, predictLoad):
    data = pd.DataFrame({"load":[load], "hpaReplica":[hpaReplica], "cpaReplica":[cpaReplica], "predictLoad":[predictLoad]})
    isNewData = False
    try:
        df = pd.read_csv('./data.csv', header=None)
        print("Data size is {size}".format(size = df.shape[0]))

    except:
        data.to_csv('./data.csv', mode="w", index=False, header=False)
        df = pd.read_csv('./data.csv', header=None)
        isNewData = True

    if not isNewData:
        df.dropna(inplace = True)
        data.to_csv('./data.csv', mode="a", index=False, header=False)


def read_data(_data_path):
    data_path = _data_path
    print("Reading pkl data...")
    input_machine = open(data_path,'rb')
    cpu_load = pickle.load(input_machine)
    cpu_load = np.array(cpu_load)

    # print(cpu_load)
    input_machine.close()
    print("Loading data...")
    return cpu_load

if __name__ == "__main__":
    data_path = "autoscaler/dataset.pkl" # 原数据集(100, 8352)

    total_seq = read_data(data_path)

    model_name = "GRU_Model_24"
    model_path = "autoscaler/" + model_name
    predict_Model = load_model(model_path)

    inputSize = int(model_name.split("_")[-1])

    random.seed(2)
    randomRow = random.randint(0,100)
    randomLoads = total_seq[randomRow]

    # 多线程用于同时执行HPA和CPA扩容逻辑
    threadPool = ThreadPoolExecutor(max_workers = 2)

    simulator = Simulator(randomLoads, predict_Model, inputSize)

    simulator.SetResources(3, 0.25, 0)
    simulator.TimedTasks(0.25)
    while(not simulator.stopFlag):
        simulator.TickTimer()
    threadPool.shutdown(wait=True)

