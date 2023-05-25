


# import modules
import os
import glob
import pandas as pd
import numpy as np

trainDataPath = r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/'
trainData = glob.glob(trainDataPath + "/*.csv")

for file in trainData:
	train_df = pd.read_csv(file, encoding='Latin-1')
	nas = train_df.isna().sum()
	print('This is the file:' + ' ' + str(file))
	print('Total N/As in their' + ' ' + str(nas))
	print(train_df.shape)