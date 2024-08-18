import pandas as pd
import numpy as np

data = pd.read_csv("D:\AI lab\exp1\exp1.csv")
print("Data:\n", data)

d = np.array(data)[:, :-1]
print("\nThe attributes are:\n", d)

target = np.array(data)[:, -1]
print("\nThe target is:\n", target)

def train(c, t):
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break

    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass

    return specific_hypothesis

final_hypothesis = train(d, target)
print("\nThe final hypothesis is:", final_hypothesis)
