import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

excel_file_path = "C:\\Users\\Advik Narendran\\ml project\\patches_gabor_15816_1 3.csv"
df = pd.read_csv(excel_file_path)

bad=df[df['class'] == 'bad']
bad=bad.drop("ImageName", axis="columns")
bad=bad.drop("class",axis="columns")

med=df[df['class'] == 'medium']
med=med.drop("ImageName", axis="columns")
med=med.drop("class",axis="columns")
print(bad)

bmean=np.mean(bad,axis=0)
bstd=np.std(bad,axis=0)

mmean=np.mean(med,axis=0)
mstd=np.std(med,axis=0)