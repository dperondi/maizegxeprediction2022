#%% import modules
import os
import glob
import pandas as pd
import numpy as np
 

#%% Get data file names
trainDataPath = r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/data/raw/Training_Data'
testDataPath = r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/data/raw/Testing_Data'
trainData = glob.glob(trainDataPath + "/*.csv")
testData = glob.glob(testDataPath + '/*.csv')

#%% Create empty lists to hold future dataframes
trainDFs = []
testDFs = []
yieldDF = []
submissionDF = []
hybridDF = []

#%% Define a numerical imputer for data
def impute_numerical(df, categorical_column, numerical_column):
    frames = []
    for i in list(set(df[categorical_column])):

        df_category = df[df[categorical_column] == i]
      
        if len(df_category) > 1:
            df_category[numerical_column].fillna(df_category[numerical_column].mean(),inplace = True)
            frames.append(df_category)
        else:
            df_category[numerical_column].fillna(df[numerical_column].mean(),inplace = True)
            frames.append(df_category)
    final_df = pd.concat(frames)
    return final_df

#%% Define a catagorical imputer for data
def impute_categorical(df, categorical_column1, categorical_column2):
    cat_frames = []
    for i in list(set(df[categorical_column1])):
        df_category = df[df[categorical_column1]== i]
        if len(df_category) > 1:    
            df_category[categorical_column2].fillna(df_category[categorical_column2].mode()[0],inplace = True)        
            cat_frames.append(df_category)
        else:
            df_category[categorical_column2].fillna(df[categorical_column2].mode()[0],inplace = True)
            cat_frames.append(df_category)    
    cat_df = pd.concat(cat_frames)
    return cat_df

#%% Prepare the data for imputation
def impute_prep(df):
    datatypes = df.dtypes
    totalNAs = df.isna().sum()
    
    for i in range(len(datatypes)):
        if datatypes[i] == 'object' and totalNAs[i] != 0:
            imputedTest = impute_categorical(df, 'Env', datatypes.index[i])
            df[datatypes.index[i]] = imputedTest[datatypes.index[i]]
        if datatypes[i] == 'int64' or datatypes[i] == 'float64' and totalNAs[i] != 0:
            imputedTest = impute_numerical(df, 'Env', datatypes.index[i])
            df[datatypes.index[i]] = imputedTest[datatypes.index[i]]

    return df

#%% Begin to looop through files and process them accordinlgy
for trainFile in trainData:
    if 'Trait' in trainFile: # Check the trait file in the train dataset and modify it to expand date time, drop columns, and create a data type of days between harvest
            train_df = pd.read_csv(trainFile, encoding='Latin-1')
            for i in train_df.columns:
                
                if i.startswith('Date'):
                    train_df[i] = pd.to_datetime(train_df[i])
                    train_df[str(i)+'_month'] = train_df[i].dt.month
                    train_df[str(i)+'_day'] = train_df[i].dt.day
                if i.startswith('Env'):
                    train_df['Year'] = train_df[i].str[-4:]
                    train_df[i] = train_df[i].str[:4]
                if i.find('Parent') != -1:
                    train_df.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Range') or i.startswith('Pass'):
                    train_df.drop(columns=i, axis =1, inplace = True)
                    

            train_df['Days_to_harvest'] = (train_df['Date_Harvested'] - train_df['Date_Planted']).dt.days
            
            train_df.drop(['Date_Planted', 'Date_Harvested'], axis =1, inplace = True)
            
            columnsToImpute = train_df.columns[train_df.isnull().any()]
            
            for i in columnsToImpute: #impute data missing in columns i based on hybrid
                imputedTest = impute_numerical(train_df, 'Hybrid', i)
                train_df[i] = imputedTest[i]
                
            train_df = impute_prep(train_df)

            train_df['Env' ] = train_df['Env'].astype(str) +"_"+ train_df["Year"].astype(str)
            df_imputed_final = train_df[['Hybrid','Env', 'Yield_Mg_ha']]
            yieldDF.append(df_imputed_final)
            
            open(r'//Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/training_trait_imputed.csv', 'w')
            df_imputed_final.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/training_trait_imputed.csv', encoding='utf-8')
            trainDFs.append(train_df)
              
    for testFile in testData: 
        if 'Weather' in trainFile and 'Weather' in testFile: # Process the weather data, impute needed values, drop certain months, and resample to make data weekly/monthly/yearly
            
            train_df = pd.read_csv(trainFile, encoding='latin-1')
            test_df = pd.read_csv(testFile, encoding='latin-1')
            
            commonWeather = train_df.columns.intersection(test_df.columns)
            
            commonCols = []
            for i in commonWeather:
                commonCols.append(i)
                
            common_train_DF = train_df.loc[:,commonCols]
            common_test_DF = test_df.loc[:,commonCols]
            
            for i in common_train_DF.columns:
                if i.startswith('Issue') or i.startswith('Comments'):
                    common_train_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Env'):
                    common_train_DF['Year'] = common_train_DF[i].str[-4:]
                    common_train_DF['Env'] = common_train_DF[i].str[:4]
                if i.startswith('LabID'):
                    common_train_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Date'):
                    common_train_DF[i] = pd.to_datetime(train_df[i], format='%Y%m%d', errors='coerce')
                    common_train_DF[str(i)+'_month'] = common_train_DF[i].dt.month
                    common_train_DF[str(i)+'_day'] = common_train_DF[i].dt.day
                        
            for i in common_test_DF.columns:
                if i.startswith('Issue') or i.startswith('Comments'):
                    common_test_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Env'):
                    common_test_DF['Year'] = common_test_DF[i].str[-4:]
                    common_test_DF['Env'] = common_test_DF[i].str[:4]
                if i.startswith('LabID'):
                    common_test_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Date'):
                    common_test_DF[i] = pd.to_datetime(common_test_DF[i], format='%Y%m%d', errors='coerce')
                    common_test_DF[str(i)+'_month'] = common_test_DF[i].dt.month
                    common_test_DF[str(i)+'_day'] = common_test_DF[i].dt.day
            
            imputed_train_df = impute_prep(common_train_DF)
            imputed_test_df = impute_prep(common_test_DF)
            
            imputed_train_df = imputed_train_df.loc[imputed_train_df['Date_month'] != 1]
            imputed_train_df = imputed_train_df.loc[imputed_train_df['Date_month'] != 2]
            imputed_train_df = imputed_train_df.loc[imputed_train_df['Date_month'] != 11]
            imputed_train_df = imputed_train_df.loc[imputed_train_df['Date_month'] != 12]
            
            imputed_test_df = imputed_test_df.loc[imputed_test_df['Date_month'] != 1]
            imputed_test_df = imputed_test_df.loc[imputed_test_df['Date_month'] != 2]
            imputed_test_df = imputed_test_df.loc[imputed_test_df['Date_month'] != 11]
            imputed_test_df = imputed_test_df.loc[imputed_test_df['Date_month'] != 12]
            
            imputed_train_df['Env' ] = imputed_train_df['Env'].astype(str) +"_"+ imputed_train_df["Year"].astype(str)
            imputed_test_df['Env' ] = imputed_test_df['Env'].astype(str) +"_"+ imputed_test_df["Year"].astype(str)
            
            dateTime_index_train = imputed_train_df.set_index('Date')
            gb = dateTime_index_train.groupby(['Env'])

            dateTime_index_train = gb.resample('m').mean()

            dateTime_index_test = imputed_test_df.set_index('Date')
            gb = dateTime_index_test.groupby(['Env'])

            dateTime_index_test = gb.resample('m').mean()
            
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/weather_weekly_train_imputed.csv', 'w')
            dateTime_index_train.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/weather_weekly_train_imputed.csv', encoding='utf-8')
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/weather_weekly_test_imputed.csv', 'w')
            dateTime_index_test.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/weather_weekly_test_imputed.csv', encoding='utf-8')
    

        if 'Soil' in trainFile and 'Soil' in testFile: # Process the soil file to remove certain columns, break out time between recieved and reported data, impute 

            train_df = pd.read_csv(trainFile, encoding='latin-1')
            test_df = pd.read_csv(testFile, encoding='latin-1')
            commonWeather = train_df.columns.intersection(test_df.columns)
            
            commonCols = []
            for i in commonWeather:
                commonCols.append(i)
                
            common_train_DF = train_df.loc[:,commonCols]
            common_test_DF = test_df.loc[:,commonCols]
            
            for i in common_train_DF.columns:
                if i.startswith('Issue') or i.startswith('Comments'):
                    common_train_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Env'):
                    common_train_DF[i] = common_train_DF[i].str[:4]
                if i.startswith('LabID'):
                    common_train_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Date'):
                    common_train_DF[i] = pd.to_datetime(train_df[i])
                    common_train_DF[str(i)+'_month'] = common_train_DF[i].dt.month
                    common_train_DF[str(i)+'_day'] = common_train_DF[i].dt.day
                    
            for i in common_test_DF.columns:
                if i.startswith('Issue') or i.startswith('Comments'):
                    common_test_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Env'):
                    common_test_DF[i] = common_test_DF[i].str[:4]
                if i.startswith('LabID'):
                    common_test_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Date'):
                    common_test_DF[i] = pd.to_datetime(common_test_DF[i])
                    common_test_DF[str(i)+'_month'] = common_test_DF[i].dt.month
                    common_test_DF[str(i)+'_day'] = common_test_DF[i].dt.day
            
            common_train_DF['Days_between_reciept_and_report'] = (common_train_DF['Date Reported'] - common_train_DF['Date Received']).dt.days
            common_train_DF.drop(['Date Reported', 'Date Received'], axis = 1, inplace = True)
            common_test_DF['Days_between_reciept_and_report'] = (common_test_DF['Date Reported'] - common_test_DF['Date Received']).dt.days
            common_test_DF.drop(['Date Reported', 'Date Received'], axis = 1, inplace = True)
            
            imputed_train_df = impute_prep(common_train_DF)
            imputed_test_df = impute_prep(common_test_DF)
            

            imputed_train_df['Env' ] = imputed_train_df['Env'].astype(str) +"_"+ imputed_train_df["Year"].astype(str)
            imputed_test_df['Env' ] = imputed_test_df['Env'].astype(str) +"_"+ imputed_test_df["Year"].astype(str)
            
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/soil_train_imputed.csv', 'w')
            imputed_train_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/soil_train_imputed.csv', encoding='utf-8')
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/soil_test_imputed.csv', 'w')
            imputed_test_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/soil_test_imputed.csv', encoding='utf-8')
        
            
        if 'EC' in trainFile and 'EC' in testFile: #Process the EC files

            train_df = pd.read_csv(trainFile, encoding='latin-1')
            test_df = pd.read_csv(testFile, encoding='latin-1')
            commonWeather = train_df.columns.intersection(test_df.columns)
            
            commonCols = []
            for i in commonWeather:
                commonCols.append(i)
                
            common_train_DF = train_df.loc[:,commonCols]
            common_test_DF = test_df.loc[:,commonCols]
            
            for i in common_train_DF.columns:
                if i.startswith('Env'):
                    common_train_DF['Year'] = common_train_DF[i].str[-4:]
                    common_train_DF[i] = common_train_DF[i].str[:4]
                    
            for i in common_test_DF.columns:
                if i.startswith('Env'):
                    common_test_DF['Year'] = common_test_DF[i].str[-4:]
                    common_test_DF[i] = common_test_DF[i].str[:4]
                    
            common_train_DF['Env' ] = common_train_DF['Env'].astype(str) +"_"+ common_train_DF["Year"].astype(str)
            common_test_DF['Env' ] = common_test_DF['Env'].astype(str) +"_"+ common_test_DF["Year"].astype(str)
                    
            
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/ec_train_imputed.csv', 'w')
            common_train_DF.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/ec_train_imputed.csv', encoding='utf-8')
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/ec_test_imputed.csv', 'w')
            common_test_DF.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/ec_test_imputed.csv', encoding='utf-8')


        if 'Meta' in trainFile and 'Meta' in testFile: ### Process the Meta Data file to include dropping columns, building a data column for days between weather station planting, impute, and write out

            train_df = pd.read_csv(trainFile, encoding='latin-1')
            test_df = pd.read_csv(testFile, encoding='latin-1')
            commonWeather = train_df.columns.intersection(test_df.columns)
            
            commonCols = []
            for i in commonWeather:
                commonCols.append(i)
                
            common_train_DF = train_df.loc[:,commonCols]
            common_test_DF = test_df.loc[:,commonCols]
            
            for i in common_train_DF.columns:
                if i.startswith('Issue') or i.startswith('Comments') or i.startswith('Exp') or i.startswith('Treatment') or i.startswith('Farm') or i.startswith('Field') or i.startswith('Trial') or i.startswith('Weather_Station_Serial') or i.startswith('Pre-plant_tillage') or i.startswith('System_Determining_Moisture') or i.startswith('Pounds_Needed_Soil') or i.startswith('Cardinal_') or i.startswith('Soil_Taxonomic') or i.startswith('In-season'):
                    common_train_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Date'):
                    common_train_DF[i] = pd.to_datetime(common_train_DF[i])
                    common_train_DF[str(i)+'_month'] = common_train_DF[i].dt.month
                    common_train_DF[str(i)+'_day'] = common_train_DF[i].dt.day
                if i.startswith('Env'):
                    common_train_DF[i] = common_train_DF[i].str[:4]
                    
            for i in common_test_DF.columns:
                if i.startswith('Issue') or i.startswith('Comments') or i.startswith('Exp') or i.startswith('Treatment') or i.startswith('Farm') or i.startswith('Field') or i.startswith('Trial') or i.startswith('Weather_Station_Serial') or i.startswith('Pre-plant_tillage') or i.startswith('System_Determining_Moisture') or i.startswith('Pounds_Needed_Soil') or i.startswith('Cardinal_') or i.startswith('Soil_Taxonomic') or i.startswith('In-season'):
                    common_test_DF.drop(columns=i, axis = 1, inplace = True)
                if i.startswith('Date'):
                    common_test_DF[i] = pd.to_datetime(common_test_DF[i])
                    common_test_DF[str(i)+'_month'] = common_test_DF[i].dt.month
                    common_test_DF[str(i)+'_day'] = common_test_DF[i].dt.day
                if i.startswith('Env'):
                    common_test_DF[i] = common_test_DF[i].str[:4]
            
            common_train_DF['Days_between_WS_placement_and_removal'] = (common_train_DF['Date_weather_station_removed'] - common_train_DF['Date_weather_station_placed']).dt.days
            common_train_DF.drop(['Date_weather_station_placed', 'Date_weather_station_removed'], axis = 1, inplace = True)
            common_test_DF['Days_between_WS_placement_and_removal'] = (common_test_DF['Date_weather_station_removed'] - common_test_DF['Date_weather_station_placed']).dt.days
            common_test_DF.drop(['Date_weather_station_placed', 'Date_weather_station_removed'], axis = 1, inplace = True)

            imputed_train_df = impute_prep(common_train_DF)
            imputed_test_df = impute_prep(common_test_DF)

            columnsToImpute = imputed_train_df.columns[imputed_train_df.isnull().any()]
            
            for i in columnsToImpute:
                imputed = impute_numerical(imputed_train_df, 'Year', i)
                imputed_train_df[i] = imputed[i]
            
            imputed_train_df['Env' ] = imputed_train_df['Env'].astype(str) +"_"+ imputed_train_df["Year"].astype(str)
            imputed_test_df['Env' ] = imputed_test_df['Env'].astype(str) +"_"+ imputed_test_df["Year"].astype(str)
            
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/meta_train_imputed.csv', 'w')
            imputed_train_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/meta_train_imputed.csv', encoding='utf-8')
            open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/meta_test_imputed.csv', 'w')
            imputed_test_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/testBed/meta_test_imputed.csv', encoding='utf-8')