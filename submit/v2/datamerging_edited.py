#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:44:08 2023

@author: bkamos
"""
#%%
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

### Variables to ignore during renaming
variablesToNotRename = ['Hybrid', 'Env', 'Yield_Mg_ha']
hybrid_encoder = preprocessing.LabelEncoder()

#%%
## EC processing
# EC test processing
ec_test = pd.read_csv('data/testBed/ec_test_imputed.csv')
ec_test = ec_test.drop(ec_test.columns[[0]], axis = 1)
ec_test = ec_test.rename(columns={x:y for x,y in zip(ec_test.columns,range(0,len(ec_test.columns))) if x not in variablesToNotRename})
ec_test = ec_test.rename(columns={c: 'ec_'+str(c) for c in ec_test.columns if c not in variablesToNotRename})

ec_test_dtype = ec_test.dtypes
for i in range(len(ec_test_dtype)):
    if ec_test_dtype[i] == 'float64':
        ec_test[ec_test_dtype.index[i]] = pd.to_numeric(ec_test[ec_test_dtype.index[i]], downcast='float')
    
    if ec_test_dtype[i] == 'int64':
        ec_test[ec_test_dtype.index[i]] = pd.to_numeric(ec_test[ec_test_dtype.index[i]], downcast='integer')


### EC train processing
ec_train = pd.read_csv('data//testBed/ec_train_imputed.csv')
ec_train = ec_train.drop(ec_train.columns[[0]], axis = 1)
ec_train = ec_train.drop_duplicates(subset=['Env'], keep='first')
ec_train = ec_train.rename(columns={x:y for x,y in zip(ec_train.columns,range(0,len(ec_train.columns))) if x not in variablesToNotRename})
ec_train = ec_train.rename(columns={c: 'ec_'+str(c) for c in ec_train.columns if c not in variablesToNotRename})

ec_train_dtype = ec_train.dtypes
for i in range(len(ec_train_dtype)):
    if ec_train_dtype[i] == 'float64':
        ec_train[ec_train_dtype.index[i]] = pd.to_numeric(ec_train[ec_train_dtype.index[i]], downcast='float')
    
    if ec_train_dtype[i] == 'int64':
        ec_train[ec_train_dtype.index[i]] = pd.to_numeric(ec_train[ec_train_dtype.index[i]], downcast='integer')

#%%
### Soil Processing
### Soil test processing
soil_test = pd.read_csv('data/testBed/soil_test_imputed.csv')
soil_test = soil_test.drop(soil_test.columns[[0,1]], axis = 1 )
soil_test = soil_test.rename(columns={x:y for x,y in zip(soil_test.columns,range(0,len(soil_test.columns))) if x not in variablesToNotRename})
soil_test = soil_test.rename(columns={c: 'soil_'+str(c) for c in soil_test.columns if c not in variablesToNotRename})

### Soil train processing
soil_train = pd.read_csv('data/testBed/soil_train_imputed.csv')
soil_train = soil_train.drop(soil_train.columns[[0,1]], axis = 1 )
soil_train = soil_train.drop_duplicates(subset=['Env'], keep='first')
soil_train = soil_train.rename(columns={x:y for x,y in zip(soil_train.columns,range(0,len(soil_train.columns))) if x not in variablesToNotRename})
soil_train = soil_train.rename(columns={c: 'soil_'+str(c) for c in soil_train.columns if c not in variablesToNotRename})

#%%
### Meta Processing
### Meta test processing
meta_test = pd.read_csv('data/testBed/meta_test_imputed.csv')
meta_test = meta_test.drop(meta_test.columns[[0,1]], axis = 1 )
meta_test = meta_test.rename(columns={x:y for x,y in zip(meta_test.columns,range(0,len(meta_test.columns))) if x not in variablesToNotRename})
meta_test = meta_test.rename(columns={c: 'meta_'+str(c) for c in meta_test.columns if c not in variablesToNotRename})

### Meta train processing
meta_train = pd.read_csv('data/testBed/meta_train_imputed.csv') 
meta_train = meta_train.drop(meta_train.columns[[0,1]], axis = 1 )
meta_train = meta_train.drop_duplicates(subset=['Env'], keep='first')
meta_train = meta_train.rename(columns={x:y for x,y in zip(meta_train.columns,range(0,len(meta_train.columns))) if x not in variablesToNotRename})
meta_train = meta_train.rename(columns={c: 'meta_'+str(c) for c in meta_train.columns if c not in variablesToNotRename})

#%%
### Genome Files processing
### actual Geome File
vcf_file =  pd.read_csv('data/geno_hybrids_noNA.csv')
vcf_file = vcf_file.rename(columns={x:y for x,y in zip(vcf_file.columns,range(0,len(vcf_file.columns))) if x not in variablesToNotRename})
vcf_file = vcf_file.rename(columns={c: 'genome_'+str(c) for c in vcf_file.columns if c not in variablesToNotRename})


vcf_file_dtype = vcf_file.dtypes
for i in range(len(vcf_file_dtype)):
    if vcf_file_dtype[i] == 'float64' or vcf_file_dtype[i] == 'int64':
        vcf_file[vcf_file_dtype.index[i]] = pd.to_numeric(vcf_file[vcf_file_dtype.index[i]], downcast='integer')

#%%
### Genome key
hybrids_with_vcfData_file = pd.read_csv('data/vcf_key_file.csv')
vcfkey = hybrids_with_vcfData_file.drop(hybrids_with_vcfData_file.columns[[0,2,3]], axis =1)

### Submit File processing
subfile = pd.read_csv('data/1_Submission_Template_2022.csv')
subfile = subfile.drop(subfile.columns[[2]], axis = 1)

### Hybrid file Processing
train_hybrid_data = pd.read_csv('data/testBed/training_trait_imputed.csv')
train_hybrid_data = train_hybrid_data.drop(train_hybrid_data.columns[[0]], axis = 1)

#%%
### Weather Data Processing
### Weather test processing
weather_test = pd.read_csv('data/testBed/weather_weekly_test_imputed.csv')


weather_test = weather_test.rename(columns={x:y for x,y in zip(weather_test.columns,range(0,len(weather_test.columns))) if x not in variablesToNotRename})

weather_test = weather_test.drop(weather_test.columns[[1]], axis = 1)
weather_test_env = weather_test.drop(weather_test.columns[[1]], axis = 1 )

weather_test_env = weather_test['Env'].unique()

weather_test_list = list(weather_test_env)

weather_test_env_unique = []

for i in range(len(weather_test_list)):
    weather_test_env_unique.append(weather_test[weather_test['Env'] == weather_test_list[i]])

test_flattenedDFs = []
for i in weather_test_env_unique:
    noenv = i.drop(i.columns[[0]], axis = 1)
    test_flattenedDFs.append(noenv.to_numpy().flatten())

test_truncFlatDFs = []
for i in test_flattenedDFs:
    if len(i) > 630:
        test_truncFlatDFs.append(i[:-18])
    else:
        test_truncFlatDFs.append(i)
        print('this is test less than 630')

test_dbnFrame = pd.DataFrame(np.column_stack(test_truncFlatDFs))
test_dbnTransformed = test_dbnFrame.T

test_dbnTransformed['Env'] = weather_test_list

weather_test = test_dbnTransformed.rename(columns={c: 'weather_'+str(c) for c in test_dbnTransformed.columns if c not in variablesToNotRename})

weather_test_dtypes = weather_test.dtypes
print(weather_test.dtypes)

for i in range(len(weather_test_dtypes)):
    if weather_test_dtypes[i] == 'float64':
        weather_test[weather_test_dtypes.index[i]] = pd.to_numeric(weather_test[weather_test_dtypes.index[i]], downcast='float')

print(weather_test.dtypes)
print(weather_test.shape)



### Weather Train processing ####
weather_train = pd.read_csv('data/testBed/weather_weekly_train_imputed.csv')

weather_train = weather_train.rename(columns={x:y for x,y in zip(weather_train.columns,range(0,len(weather_train.columns))) if x not in variablesToNotRename})

weather_train = weather_train.drop(weather_train.columns[[1]], axis = 1)
weather_train_env = weather_train.drop(weather_train.columns[[1]], axis = 1 )

weather_train_env = weather_train['Env'].unique()

weather_train_list = list(weather_train_env)

weather_train_env_unique = []

for i in range(len(weather_train_list)):
    weather_train_env_unique.append(weather_train[weather_train['Env'] == weather_train_list[i]])

train_flattenedDFs = []
for i in weather_train_env_unique:
    noenv = i.drop(i.columns[[0]], axis = 1)
    train_flattenedDFs.append(noenv.to_numpy().flatten())

train_truncFlatDFs = []
for i in train_flattenedDFs:
    if len(i) > 630:
        train_truncFlatDFs.append(i[:-18])
    else:
        train_truncFlatDFs.append(i)

train_dbnFrame = pd.DataFrame(np.column_stack(train_truncFlatDFs))
train_dbnTransformed = train_dbnFrame.T

train_dbnTransformed['Env'] = weather_train_list

weather_train = train_dbnTransformed.rename(columns={c: 'weather_'+str(c) for c in train_dbnTransformed.columns if c not in variablesToNotRename})

weather_train_dtypes = weather_train.dtypes
print(weather_train.dtypes)

for i in range(len(weather_train_dtypes)):
    if weather_train_dtypes[i] == 'float64':
        weather_train[weather_train_dtypes.index[i]] = pd.to_numeric(weather_train[weather_train_dtypes.index[i]], downcast='float')

print(weather_train.dtypes)
print(weather_train.shape)

#%% Merge and write the train and test data
### Merge and write test data
test_data = subfile
print(test_data.shape)
test_data = pd.merge(left=test_data, right=vcfkey, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
print(test_data.shape)
test_data = pd.merge(left=test_data, right=vcf_file, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
print(test_data.shape)
test_data = test_data.drop(columns=['train', 'test', 'vcf'])

hybrid_encoder.fit(test_data['Hybrid'])
test_data['Hybrid'] = hybrid_encoder.transform(test_data['Hybrid'])
print(test_data['Hybrid'])

test_data = pd.merge(left=test_data, right=meta_test, how='left', left_on=['Env'], right_on=['Env'])
print(test_data.shape)
test_data = pd.merge(left=test_data, right=weather_test, how='left', left_on=['Env'], right_on=['Env'])
print(test_data.shape)
test_data = pd.merge(left=test_data, right=soil_test, how='left', left_on=['Env'], right_on=['Env'])
print(test_data.shape)
test_data = pd.merge(left=test_data, right=ec_test, how='left', left_on=['Env'], right_on=['Env'])
print(test_data.shape)
test_nan0 = test_data.replace(np.nan,0)

open(r'/data/testBed/test_weekly_ec_meta_weather_soil_genome_Na0.csv', 'w')
test_nan0.to_csv(r'/data/testBed/test_weekly_ec_meta_weather_soil_genome_Na0.csv', encoding='utf-8')

### Merge and write train data
train_data = train_hybrid_data
print(train_data.shape)
train_data = pd.merge(left=train_data, right=vcfkey, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
print(train_data.shape)
train_data = pd.merge(left=train_data, right=vcf_file, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
print(train_data.shape)
train_data = train_data.drop(columns=['train', 'test', 'vcf'])
print(train_data.shape)

hybrid_encoder.fit(train_data['Hybrid'])
train_data['Hybrid'] = hybrid_encoder.transform(train_data['Hybrid'])


meta_train = meta_train.drop_duplicates(subset=['Env'], keep='first')
train_data = pd.merge(left=train_data, right=meta_train, how='left', left_on=['Env'], right_on=['Env'])
print(train_data.shape)
train_data = pd.merge(left=train_data, right=weather_train, how='left', left_on=['Env'], right_on=['Env'])
print(train_data.shape)
train_data = pd.merge(left=train_data, right=soil_train, how='left', left_on=['Env'], right_on=['Env'])
print(train_data.shape)
train_data = pd.merge(left=train_data, right=ec_train, how='left', left_on=['Env'], right_on=['Env'])
print(train_data.shape)
train_nan0 = train_data.replace(np.nan,0)
open(r'/data/testBed/train_yearly_ec_meta_weather_soil_genome_2NA0.csv', 'w')
train_nan0.to_csv(r'/data/testBed/train_yearly_ec_meta_weather_soil_genome_2NA0.csv', encoding='utf-8')
