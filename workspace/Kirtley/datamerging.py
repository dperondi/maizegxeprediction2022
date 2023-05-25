#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:44:08 2023

@author: bkamos
"""

import pandas as pd
import os


ec_test = pd.read_csv('data/ec_test_imputed.csv')
ec_train = pd.read_csv('data/ec_train_imputed.csv')
soil_test = pd.read_csv('data/soil_test_imputed.csv')
soil_train = pd.read_csv('data/soil_train_imputed.csv')
meta_test = pd.read_csv('data/meta_test_imputed.csv')
meta_train = pd.read_csv('data/meta_train_imputed.csv') 
weather_test = pd.read_csv('data/weather_test_imputed.csv')
weather_train = pd.read_csv('data/weather_train_imputed.csv')
vcf_file =  pd.read_csv('data/geno_hybrids_1.csv')
hybrids_with_vcfData_file = pd.read_csv('data/vcf_key_file.csv')
subfile = pd.read_csv('data/1_Submission_Template_2022.csv')
train_hybrid_data = pd.read_csv('data/training_trait_imputed.csv')


vcfkey = hybrids_with_vcfData_file.drop(hybrids_with_vcfData_file.columns[[0]], axis =1)
subfile = subfile.drop(subfile.columns[[2]], axis = 1)


variablesToNotRename = ['Hybrid', 'Env', 'Yield_Mg_ha']


ec_test = ec_test.rename(columns={c: 'ec_'+c for c in ec_test.columns if c not in variablesToNotRename})
soil_test = soil_test.rename(columns={c: 'soil_'+c for c in soil_test.columns if c not in variablesToNotRename})
meta_test = meta_test.rename(columns={c: 'meta_'+c for c in meta_test.columns if c not in variablesToNotRename})
weather_test = weather_test.rename(columns={c: 'weather_'+c for c in weather_test.columns if c not in variablesToNotRename})
vcf_file = vcf_file.rename(columns={c: 'genome_'+c for c in weather_test.columns if c not in variablesToNotRename})


ec_test = ec_test.drop(ec_test.columns[[0]], axis = 1)
soil_test = soil_test.drop(soil_test.columns[[0,1]], axis = 1 )
meta_test = meta_test.drop(meta_test.columns[[0,1]], axis = 1 )
weather_test = weather_test.drop(weather_test.columns[[1]], axis = 1 )


ec_train = ec_train.rename(columns={c: 'ec_'+c for c in ec_train.columns if c not in variablesToNotRename})
soil_train = soil_train.rename(columns={c: 'soil_'+c for c in soil_train.columns if c not in variablesToNotRename})
meta_train = meta_train.rename(columns={c: 'meta_'+c for c in meta_train.columns if c not in variablesToNotRename})
weather_train = weather_train.rename(columns={c: 'weather_'+c for c in weather_train.columns if c not in variablesToNotRename})


ec_train = ec_train.drop(ec_train.columns[[0]], axis = 1)
soil_train = soil_train.drop(soil_train.columns[[0,1]], axis = 1 )
meta_train = meta_train.drop(meta_train.columns[[0,1]], axis = 1 )
weather_train = weather_train.drop(weather_train.columns[[1]], axis = 1 )


test_data = subfile
test_data = pd.merge(left=test_data, right=vcfkey, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
test_data = pd.merge(left=test_data, right=vcf_file, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
test_data = test_data.drop(columns=['train', 'test', 'vcf'])
test_data = pd.merge(left=test_data, right=meta_test, how='left', left_on=['Env'], right_on=['Env'])
test_data = pd.merge(left=test_data, right=weather_test, how='left', left_on=['Env'], right_on=['Env'])
test_data = pd.merge(left=test_data, right=soil_test, how='left', left_on=['Env'], right_on=['Env'])
test_data = pd.merge(left=test_data, right=ec_test, how='left', left_on=['Env'], right_on=['Env'])

open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/test_ec_meta_weather_soil_genome.csv', 'w')
test_data.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/test_ec_meta_weather_soil_genome.csv', encoding='utf-8')


train_data = train_hybrid_data
train_data = pd.merge(left=train_data, right=vcfkey, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
train_data = pd.merge(left=train_data, right=vcf_file, how='left', left_on=['Hybrid'], right_on = ['Hybrid'])
train_data = train_data.drop(columns=['train', 'test', 'vcf'])
train_data = pd.merge(left=train_data, right=meta_train, how='left', left_on=['Env'], right_on=['Env'])
train_data = pd.merge(left=train_data, right=weather_train, how='left', left_on=['Env'], right_on=['Env'])
train_data = pd.merge(left=train_data, right=soil_train, how='left', left_on=['Env'], right_on=['Env'])
train_data = pd.merge(left=train_data, right=ec_train, how='left', left_on=['Env'], right_on=['Env'])

open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/train_ec_meta_weather_soil_genome.csv', 'w')
test_data.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/train_ec_meta_weather_soil_genome.csv', encoding='utf-8')


# ec_soil_train = pd.merge(ec_train, soil_train, on ='Env')
# ec_soil_train = pd.merge(ec_train, soil_train, on = ['Env', 'Hybrid'])
# ec_soil_meta_train = pd.merge(ec_soil_train, meta_train, on = ['Env', 'Hybrid'])
# ec_soil_meta_weather_train = pd.merge(ec_soil_meta_train, weather_train, on = ['Env', 'Hybrid']) 
# test_dfs_hyb_yeild_drop = [soil_test, meta_test, weather_test]

# for i in test_dfs_hyb_yeild_drop:
#     df = i.drop(i.columns[[1,3]], axis = 1)
    
    
# test_dfs = [ec_test, soil_test, meta_test, weather_test]
# train_dfs = [ec_train, soil_train, meta_train, weather_train]

# no_unnamed_test = []
# no_unnamed_train = []

# for i in test_dfs:
#     print(i)
#     df = i.drop(i.columns[[0]], axis = 1)
#     no_unnamed_test.append(df)
    
# for i in train_dfs:
#     df = i.drop(i.columns[[0]], axis = 1)
#     no_unnamed_train.append(df)
    
