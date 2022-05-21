import pandas as pd
from matplotlib import pyplot as plt


try:
    df = pd.read_csv('./data.csv', header=None)
    realLoad = df[0].tolist()
    expectedReplica = df[1].tolist()
    hpaReplica = df[2].tolist()
    cpaReplica = df[3].tolist()
    predictLoad = df[4].tolist()
except:
    print("read Data failed!!!")

real = []
for i in range(len(realLoad)):
    real.append(realLoad[i] * 12)

predict = []
for i in range(len(predictLoad)):
    predict.append(predictLoad[i] * 12)

plt.figure()
plt.grid(True)

plt.plot(real, label="Real Load", color="red", ls = "dashed")
# plt.plot(predict, label="Predict Load")
plt.plot(hpaReplica, label="HPA Replicas", color="green", lw=3)
plt.plot(cpaReplica, label="CPA Replicas", color="blue", lw=3)
# plt.plot(expectedReplica, label="Expected Replicas")

for y in range(0, 7, 1):
    a = '%.2f' % (y/12)
    a = str(a)
    plt.text(210, y, a)

plt.xlabel("Time Points(15s)", fontsize=20) 
plt.ylabel("Replicas", fontsize=20)
plt.text(220, 20, 'Load', rotation = 90)
plt.legend(loc=1,labelspacing=2,handlelength=3,fontsize=10,shadow=True)
plt.twinx()
plt.ylabel("Load", fontsize=20)
plt.tick_params(axis='y',color='w')
plt.yticks(color='w')
plt.show()