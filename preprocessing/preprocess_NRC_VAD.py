import pickle
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("./data/kgs/NRC-VAD-Lexicon.txt", sep='\t')

    NRC = {}
    for i, row in df.iterrows():
        if len(NRC) !=i:
            print(i)
        NRC[row[0]] = tuple(row[1:])

    with open("./data/KB/NRC.pkl", "wb") as f:
        pickle.dump(NRC,f)