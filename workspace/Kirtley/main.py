# TODO

# Ensure files are printed correctly with the correct column names - file name is #_Meta_train.csv while col name is X_train
# create a test and train file and ensure # features is equal when all merged - just merge at end across test and train - works across all train data if yearly average is calculated for weather
# Do we need to make the catagorical variables numerical?
# Drop/Check the date/time for planting in testing data - code writen to remove weather data from Jan, feb, and past dec 14 in train data and from jan, feb, Dec, and before 3/14 in test
# Add yeild data to each environment within every dataframe that i generate (year_env yield hybrid - each frame)
#### If we want to combine across all frames we need to average weather data by year
### Add hybrid data where VCF = true



# import modules
import os
import glob
import pandas as pd
import numpy as np
 

# Get data file names
trainDataPath = r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/data/raw/Training_Data'
testDataPath = r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/data/raw/Testing_Data'
trainData = glob.glob(trainDataPath + "/*.csv")
testData = glob.glob(testDataPath + '/*.csv')

# Empty lists
trainDFs = []
testDFs = []
yieldDF = []
submissionDF = []
hybridDF = []

def impute_numerical(df, categorical_column, numerical_column):
    frames = []
    for i in list(set(df[categorical_column])):
        # print('This is i' + str(i))
        df_category = df[df[categorical_column] == i]
        # print('this is df cat' + str(df_category))
        # print(len(df_category))
        if len(df_category) > 1:
            df_category[numerical_column].fillna(df_category[numerical_column].mean(),inplace = True)
            frames.append(df_category)
        else:
            df_category[numerical_column].fillna(df[numerical_column].mean(),inplace = True)
            frames.append(df_category)
    final_df = pd.concat(frames)
    return final_df


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

def impute_prep(df):
    datatypes = df.dtypes
    # print(datatypes)
    totalNAs = df.isna().sum()
    # print(totalNAs)
    # print(testDtypes.index)
    
    for i in range(len(datatypes)):
        if datatypes[i] == 'object' and totalNAs[i] != 0:
            # print(i)
            # print(testDtypes.index[i])
            imputedTest = impute_categorical(df, 'Env', datatypes.index[i])
            df[datatypes.index[i]] = imputedTest[datatypes.index[i]]
        if datatypes[i] == 'int64' or datatypes[i] == 'float64' and totalNAs[i] != 0:
            imputedTest = impute_numerical(df, 'Env', datatypes.index[i])
            df[datatypes.index[i]] = imputedTest[datatypes.index[i]]

    return df


for trainFile in trainData:
    if 'Trait' in trainFile:
            train_df = pd.read_csv(trainFile, encoding='Latin-1')
            for i in train_df.columns:
                
                # print(i)
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
                # if i.startswith('Plot'):
                #     train_df = train_df[i]
                    
            
# Using + operator to combine two columns
            # train_df['Env' ] = train_df['Env'].astype(str) +"_"+ train_df["Year"].astype(str)

            train_df['Days_to_harvest'] = (train_df['Date_Harvested'] - train_df['Date_Planted']).dt.days
            
            # print(train_df.isna().sum().sum())
            # train_df = train_df[train_df['Plot_Area_ha'].notna()] ### This missing NA could be imputed
            train_df.drop(['Date_Planted', 'Date_Harvested'], axis =1, inplace = True)
            
            # print(train_df.isna().sum())
            
            
            columnsToImpute = train_df.columns[train_df.isnull().any()]
            
            for i in columnsToImpute:
                imputedTest = impute_numerical(train_df, 'Hybrid', i)
                train_df[i] = imputedTest[i]
                
            train_df = impute_prep(train_df)

            # print(train_df.isna().sum())
            train_df['Env' ] = train_df['Env'].astype(str) +"_"+ train_df["Year"].astype(str)
            # imputed_train_df = impute_prep(train_df)
            df_imputed_final = train_df[['Hybrid','Env', 'Yield_Mg_ha']]
            yieldDF.append(df_imputed_final)
            
            
            # train_df = train_df.add_suffix('_Trait')
     
            # print(train_df)
            # print(df_imputed_final)
            
            open(r'//Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/training_trait_imputed.csv', 'w')
            df_imputed_final.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/training_trait_imputed.csv', encoding='utf-8')
            # trainDFs.append(train_df)
              
#     for testFile in testData:
        
#     #     # print(testFile)
#     #     # print(trainFile)
#         if 'Weather' in trainFile and 'Weather' in testFile: ### Done and imputed across all
            
#             train_df = pd.read_csv(trainFile, encoding='latin-1')
#             test_df = pd.read_csv(testFile, encoding='latin-1')
            
#             # print(train_df)
#             # print(test_df)
#             commonWeather = train_df.columns.intersection(test_df.columns)
#             # print(commonWeather)
            
#             commonCols = []
#             for i in commonWeather:
#                 commonCols.append(i)
                
#             common_train_DF = train_df.loc[:,commonCols]
#             common_test_DF = test_df.loc[:,commonCols]
            
#             for i in common_train_DF.columns:
#                 if i.startswith('Issue') or i.startswith('Comments'):
#                     common_train_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Env'):
#                     common_train_DF['Year'] = common_train_DF[i].str[-4:]
#                     common_train_DF['Env'] = common_train_DF[i].str[:4]
#                 if i.startswith('LabID'):
#                     common_train_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Date'):
#                     common_train_DF[i] = pd.to_datetime(train_df[i], format='%Y%m%d', errors='coerce')
#                     common_train_DF[str(i)+'_month'] = common_train_DF[i].dt.month
#                     common_train_DF[str(i)+'_day'] = common_train_DF[i].dt.day
                        
#             for i in common_test_DF.columns:
#                 if i.startswith('Issue') or i.startswith('Comments'):
#                     common_test_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Env'):
#                     common_test_DF['Year'] = common_test_DF[i].str[-4:]
#                     common_test_DF['Env'] = common_test_DF[i].str[:4]
#                 if i.startswith('LabID'):
#                     common_test_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Date'):
#                     # print(i)
#                     common_test_DF[i] = pd.to_datetime(common_test_DF[i], format='%Y%m%d', errors='coerce')
#                     common_test_DF[str(i)+'_month'] = common_test_DF[i].dt.month
#                     common_test_DF[str(i)+'_day'] = common_test_DF[i].dt.day

            
            
#             # print(common_train_monthly)
#             # print(common_train_monthly.describe)
#             # print(common_train_monthly.shape)

#             # common_train_DF.drop(['Date'], axis = 1, inplace = True)
#             # common_test_DF.drop(['Date'], axis = 1, inplace = True)
            
#             imputed_train_df = impute_prep(common_train_DF)
#             imputed_test_df = impute_prep(common_test_DF)
            

#             # print(common_train_DF.shape)

            
#             # print(common_test_DF.isna().sum())
#             # print(imputed_test_df.isna().sum())
#             # print(imputed_test_df.shape)

#             # print(common_train_DF)
#             imputed_train_df = imputed_train_df.loc[imputed_train_df['Date_month'] != 1]
#             # print(common_train_DF.shape)
#             imputed_train_df = imputed_train_df.loc[imputed_train_df['Date_month'] != 2]
#             # print(common_train_DF.shape)
#             pastHavestingIndex = imputed_train_df[(imputed_train_df['Date_month'] == 12) & (imputed_train_df['Date_day'] >= 14)].index
#             # pastPlantingData = common_train_DF[(common_train_DF['Date_month'] == 12) & (common_train_DF['Date_day'] >= 14)]
#             imputed_train_df.drop(pastHavestingIndex, inplace = True)
            
            
#             imputed_test_df = imputed_test_df.loc[imputed_test_df['Date_month'] != 1]
#             # print(common_test_DF.shape)
#             imputed_test_df = imputed_test_df.loc[imputed_test_df['Date_month'] != 2]
#             # print(common_test_DF.shape)
#             imputed_test_df = imputed_test_df.loc[imputed_test_df['Date_month'] != 12]
#             # print(common_test_DF.shape)
#             beforePlantingIndex = imputed_test_df[(imputed_test_df['Date_month'] == 3) & (imputed_test_df['Date_day'] <= 14)].index
#             imputed_test_df.drop(beforePlantingIndex, inplace = True)
#             # print(imputed_test_df.shape)
          
#             # print(common_test_DF.shape)
            
#             # print(common_test_DF)
            

#             # print(imputed_train_df.isna().sum())
#             # print(imputed_test_df.isna().sum())
            
#             imputed_train_df['Env' ] = imputed_train_df['Env'].astype(str) +"_"+ imputed_train_df["Year"].astype(str)
#             imputed_test_df['Env' ] = imputed_test_df['Env'].astype(str) +"_"+ imputed_test_df["Year"].astype(str)
            
#             dateTime_index_train = imputed_train_df.set_index('Date')
#             gb = dateTime_index_train.groupby(['Env'])

#             # print(dateTime_index_train)

#             dateTime_index_train = gb.resample('W').mean()

#             # print(dateTime_index_train)

#             dateTime_index_test = imputed_test_df.set_index('Date')
#             gb = dateTime_index_test.groupby(['Env'])

#             # print(dateTime_index_test)

#             dateTime_index_test = gb.resample('W').mean()
            
#             # dateTime_index_train.drop(['Date'], axis = 1, inplace = True)
#             # dateTime_index_test.drop(['Date'], axis = 1, inplace = True)

#             # print(dateTime_index_test)
#             # combinedDF = pd.merge(imputed_train_df, imputed_test_df, on='Env')
            
#             # print(common_train_DF.isna().sum())
#             # print(common_test_DF.isna().sum())
#             # print(common_test_DF.describe())
            
#             # common_train_DF = common_train_DF.add_suffix('_weather')
#             # common_test_DF = common_test_DF.add_suffix('_weather')
            
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/weather_train_imputed.csv', 'w')
#             dateTime_index_train.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/weather_train_imputed.csv', encoding='utf-8')
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/weather_test_imputed.csv', 'w')
#             dateTime_index_test.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/weather_test_imputed.csv', encoding='utf-8')
            
#             # trainDFs.append(dateTime_index_train)
#             # testDFs.append(dateTime_index_test)

#         if 'Soil' in trainFile and 'Soil' in testFile: ### Impute catagorical (texture) and all remaining numerical columns - this is done and a bit fun with imputing based on datatypes within the rows
#             train_df = pd.read_csv(trainFile, encoding='latin-1')
#             test_df = pd.read_csv(testFile, encoding='latin-1')
#             commonWeather = train_df.columns.intersection(test_df.columns)
            
#             commonCols = []
#             for i in commonWeather:
#                 commonCols.append(i)
                
#             common_train_DF = train_df.loc[:,commonCols]
#             common_test_DF = test_df.loc[:,commonCols]
            
#             for i in common_train_DF.columns:
#                 if i.startswith('Issue') or i.startswith('Comments'):
#                     common_train_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Env'):
#                     common_train_DF[i] = common_train_DF[i].str[:4]
#                 if i.startswith('LabID'):
#                     common_train_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Date'):
#                     common_train_DF[i] = pd.to_datetime(train_df[i])
#                     common_train_DF[str(i)+'_month'] = common_train_DF[i].dt.month
#                     common_train_DF[str(i)+'_day'] = common_train_DF[i].dt.day
                    
#             for i in common_test_DF.columns:
#                 if i.startswith('Issue') or i.startswith('Comments'):
#                     common_test_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Env'):
#                     common_test_DF[i] = common_test_DF[i].str[:4]
#                 if i.startswith('LabID'):
#                     common_test_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Date'):
#                     # print(i)
#                     common_test_DF[i] = pd.to_datetime(common_test_DF[i])
#                     common_test_DF[str(i)+'_month'] = common_test_DF[i].dt.month
#                     common_test_DF[str(i)+'_day'] = common_test_DF[i].dt.day
            
#             common_train_DF['Days_between_reciept_and_report'] = (common_train_DF['Date Reported'] - common_train_DF['Date Received']).dt.days
#             common_train_DF.drop(['Date Reported', 'Date Received'], axis = 1, inplace = True)
#             common_test_DF['Days_between_reciept_and_report'] = (common_test_DF['Date Reported'] - common_test_DF['Date Received']).dt.days
#             common_test_DF.drop(['Date Reported', 'Date Received'], axis = 1, inplace = True)
            
           
#             # print(common_train_DF['Year'])
#             # print(common_train_DF['Env'])
            
#             # print(common_train_DF.isna().sum())
#             # print(common_test_DF.isna().sum())
#             imputed_train_df = impute_prep(common_train_DF)
#             imputed_test_df = impute_prep(common_test_DF)
            
#             # print(imputed_train_df.isna().sum())
#             # print(imputed_test_df.isna().sum())

#             imputed_train_df['Env' ] = imputed_train_df['Env'].astype(str) +"_"+ imputed_train_df["Year"].astype(str)
#             imputed_test_df['Env' ] = imputed_test_df['Env'].astype(str) +"_"+ imputed_test_df["Year"].astype(str)
#             # combinedDF = pd.merge(imputed_train_df, imputed_test_df, on='Env')
            
            
            
#         #     common_train_DF = common_train_DF.add_suffix('soil_trait')
#         #     common_test_DF = common_test_DF.add_suffix('soil_test')
            
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/soil_train_imputed.csv', 'w')
#             imputed_train_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/soil_train_imputed.csv', encoding='utf-8')
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/soil_test_imputed.csv', 'w')
#             imputed_test_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/soil_test_imputed.csv', encoding='utf-8')
        
#             # combinedDFs.append(combinedDF)
#             # trainDFs.append(imputed_train_df)
#             # testDFs.append(imputed_test_df)
        
        
        
            
#         if 'EC' in trainFile and 'EC' in testFile: #### EC features are now identical across both test and training data
#             train_df = pd.read_csv(trainFile, encoding='latin-1')
#             test_df = pd.read_csv(testFile, encoding='latin-1')
#             commonWeather = train_df.columns.intersection(test_df.columns)
            
#             commonCols = []
#             for i in commonWeather:
#                 commonCols.append(i)
                
#             common_train_DF = train_df.loc[:,commonCols]
#             common_test_DF = test_df.loc[:,commonCols]
            
#             for i in common_train_DF.columns:
#                 if i.startswith('Env'):
#                     # print(i)
#                     common_train_DF['Year'] = common_train_DF[i].str[-4:]
#                     common_train_DF[i] = common_train_DF[i].str[:4]
                    
#             for i in common_test_DF.columns:
#                 if i.startswith('Env'):
#                     # print(i)
#                     common_test_DF['Year'] = common_test_DF[i].str[-4:]
#                     common_test_DF[i] = common_test_DF[i].str[:4]
                    
#             common_train_DF['Env' ] = common_train_DF['Env'].astype(str) +"_"+ common_train_DF["Year"].astype(str)
#             common_test_DF['Env' ] = common_test_DF['Env'].astype(str) +"_"+ common_test_DF["Year"].astype(str)
                    
#             # common_train_DF = common_train_DF.add_suffix('EC_trait')
#             # common_test_DF = common_test_DF.add_suffix('EC_test')
            
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/ec_train_imputed.csv', 'w')
#             common_train_DF.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/ec_train_imputed.csv', encoding='utf-8')
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/ec_test_imputed.csv', 'w')
#             common_test_DF.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/ec_test_imputed.csv', encoding='utf-8')
        
#             # combinedDF = pd.merge(common_train_DF, common_test_DF, on='Env')
            
            
#             # combinedDFs.append(combinedDF)
#             # trainDFs.append(common_train_DF)
#             # testDFs.append(common_test_DF)



#         if 'Meta' in trainFile and 'Meta' in testFile: ### I think that this is done - drop date planted data
#             train_df = pd.read_csv(trainFile, encoding='latin-1')
#             test_df = pd.read_csv(testFile, encoding='latin-1')
#             commonWeather = train_df.columns.intersection(test_df.columns)
            
#             commonCols = []
#             for i in commonWeather:
#                 commonCols.append(i)
                
#             common_train_DF = train_df.loc[:,commonCols]
#             common_test_DF = test_df.loc[:,commonCols]
            
#             for i in common_train_DF.columns:
#                 if i.startswith('Issue') or i.startswith('Comments') or i.startswith('Exp') or i.startswith('Treatment') or i.startswith('Farm') or i.startswith('Field') or i.startswith('Trial') or i.startswith('Weather_Station_Serial') or i.startswith('Pre-plant_tillage') or i.startswith('System_Determining_Moisture') or i.startswith('Pounds_Needed_Soil') or i.startswith('Cardinal_') or i.startswith('Soil_Taxonomic') or i.startswith('In-season'):
#                     common_train_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Date'):
#                     common_train_DF[i] = pd.to_datetime(common_train_DF[i])
#                     common_train_DF[str(i)+'_month'] = common_train_DF[i].dt.month
#                     common_train_DF[str(i)+'_day'] = common_train_DF[i].dt.day
#                 if i.startswith('Env'):
#                     common_train_DF[i] = common_train_DF[i].str[:4]
                    
#             for i in common_test_DF.columns:
#                 if i.startswith('Issue') or i.startswith('Comments') or i.startswith('Exp') or i.startswith('Treatment') or i.startswith('Farm') or i.startswith('Field') or i.startswith('Trial') or i.startswith('Weather_Station_Serial') or i.startswith('Pre-plant_tillage') or i.startswith('System_Determining_Moisture') or i.startswith('Pounds_Needed_Soil') or i.startswith('Cardinal_') or i.startswith('Soil_Taxonomic') or i.startswith('In-season'):
#                     common_test_DF.drop(columns=i, axis = 1, inplace = True)
#                 if i.startswith('Date'):
#                     common_test_DF[i] = pd.to_datetime(common_test_DF[i])
#                     common_test_DF[str(i)+'_month'] = common_test_DF[i].dt.month
#                     common_test_DF[str(i)+'_day'] = common_test_DF[i].dt.day
#                 if i.startswith('Env'):
#                     common_test_DF[i] = common_test_DF[i].str[:4]
            
#             common_train_DF['Days_between_WS_placement_and_removal'] = (common_train_DF['Date_weather_station_removed'] - common_train_DF['Date_weather_station_placed']).dt.days
#             common_train_DF.drop(['Date_weather_station_placed', 'Date_weather_station_removed'], axis = 1, inplace = True)
#             common_test_DF['Days_between_WS_placement_and_removal'] = (common_test_DF['Date_weather_station_removed'] - common_test_DF['Date_weather_station_placed']).dt.days
#             common_test_DF.drop(['Date_weather_station_placed', 'Date_weather_station_removed'], axis = 1, inplace = True)
            
            
#             # print(common_train_DF.isna().sum())
#             # print(common_test_DF.isna().sum())

#             imputed_train_df = impute_prep(common_train_DF)
#             imputed_test_df = impute_prep(common_test_DF)

#             columnsToImpute = imputed_train_df.columns[imputed_train_df.isnull().any()]
            
#             for i in columnsToImpute:
#                 imputed = impute_numerical(imputed_train_df, 'Year', i)
#                 imputed_train_df[i] = imputed[i]

#             # print(imputed_train_df.isna().sum())
#             # print(imputed_test_df.isna().sum())
            
#             imputed_train_df['Env' ] = imputed_train_df['Env'].astype(str) +"_"+ imputed_train_df["Year"].astype(str)
#             imputed_test_df['Env' ] = imputed_test_df['Env'].astype(str) +"_"+ imputed_test_df["Year"].astype(str)
            
#             # combinedDF = pd.merge(imputed_train_df, imputed_test_df, on='Env')
            
            
#             # common_train_DF = common_train_DF.add_suffix('meta_trait')
#             # common_test_DF = common_test_DF.add_suffix('meta_test')
            
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/meta_train_imputed.csv', 'w')
#             imputed_train_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/meta_train_imputed.csv', encoding='utf-8')
#             open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/meta_test_imputed.csv', 'w')
#             imputed_test_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/meta_test_imputed.csv', encoding='utf-8')
        
#             # combinedDFs.append(combinedDF)
#             # trainDFs.append(imputed_train_df)
#             # testDFs.append(imputed_test_df)
            
#             # trainDFs.append(common_train_DF)
#             # testDFs.append(common_test_DF)

# # for filepath in testData:
# #     if 'Submission' in filepath:
# #         sub_df = pd.read_csv(filepath, encoding='Latin-1')
# #         submissionDF.append(sub_df)
        
# for filepath in trainData:
#     if 'names' in filepath:
#         hybrid_df = pd.read_csv(filepath, encoding = 'Latin-1')
#         hybrid_df = hybrid_df[hybrid_df['vcf'] == True]
#         open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/vcf_file.csv', 'w')
#         hybrid_df.to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/vcf_file.csv', encoding='utf-8')
        # hybridDF.append(hybrid_df)
        
        

# print(trainDFs[0])

#### Creation of merged yeild data and training and test data on Env = ENV_year
# trainDF1 = pd.merge(yieldDF[0], trainDFs[0], on ='Env')
# trainDF2 = pd.merge(yieldDF[0], trainDFs[1], on ='Env')
# trainDF3 = pd.merge(yieldDF[0], trainDFs[2], on ='Env')
# trainDF4 = pd.merge(yieldDF[0], trainDFs[3], on ='Env')

# testDF1 = pd.merge(left = submissionDF[0], right=hybridDF[0], how = 'left', left_on =['Hybrid'], right_on = ['Hybrid'])
# testDF1 = pd.merge(left = testDF1, right= testDFs[1], on ='Env')
# testDF3 = pd.merge(submissionDF[0], testDFs[2], on ='Env')
# testDF4 = pd.merge(submissionDF[0], testDFs[3], on ='Env')

# dfsToWrite = [trainDF1, testDF1]
# script = ['weather_train_monthly', 'weather_test_monthly']

# for i in range(len(dfsToWrite)):
#     open(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/'+ script[i] +'_imputed.csv', 'w')
#     dfsToWrite[i].to_csv(r'/Users/bkamos/Documents/GitHub/maizegxeprediction2022/workspace/Kirtley/data/' + script[i] + '_imputed.csv', encoding='utf-8')





# print(merged1.head())
# print(merged1.shape)
# print(merged1.isna().sum())
# merged2 = pd.merge(merged1, trainDFs[1], on = 'Env')
# print(merged2.shape)
# print(merged2.isna().sum())
# merged3 = pd.merge(merged2, trainDFs[2], on = 'Env')
# print(merged3.shape)
# print(merged2.isna().sum())
# merged4 = pd.merge(merged2, trainDFs[3], on = 'Env')
# print(merged4.shape)
# print(merged4.isna().sum())
# merged3 = pd.merge(merged2, combinedDFs[3], on = 'Env')

# print(merged3.shape())
# print(test.isna().sum())
        




# The below code was the original body of code that was used to impute across a dataframe

# 





















        
   #     print(train_df.columns.intersection(test_df.columns))
        #     # print(train_df.columns.difference(test_df.columns))
        #     # print(test_df.columns.difference(train_df.columns))



# for file in trainData
            
#     # print(filename)
#     individual_df = pd.read_csv(filename, encoding='latin-1')
#     # print(individual_df.shape)
#     naNColumnRemoval = individual_df[individual_df.columns[individual_df.isnull().mean() < .25]]
#     # test = naNColumnRemoval.isnull().mean()
#     # test1 = naNColumnRemoval.isnull().mean(axis = 1)
#     # print(test)
#     # print(test1)
    
#     rowNanRemoval = naNColumnRemoval.dropna()
#     # print(rowNanRemoval)
    
#     corr_matrix = rowNanRemoval.corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#     # print(upper)
#     to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#     # print(to_drop)
#     test = rowNanRemoval.drop(rowNanRemoval[to_drop], axis = 1)
#     # print(test.shape)
#     # test = rowNanRemoval.isnull().mean()
#     # test1 = rowNanRemoval.isnull().mean(axis = 1)
#     # print(test)
#     # print(test1)
#     # # print(test)
    
#     dfs.append(test)
#######

# nullsRemoved = []
# for i in dfs:
#     test = i.isnull().mean()
#     # print(test)
### This was an attempt to create numerical values out of catagorical ones (could be applied but in a different way)
# dummieCheck = []
# for i in dfs:
#     dummieCheck.append(pd.get_dummies(i))
    # print(i.dtypes)
    
    
### Gets headers and puts them in a list of lists 
# headersLoL = []
# for i in dfs:
#  	headersLoL.append(i.columns.values.tolist())

# headers = []
# for i in headersLoL:
#  	for j in i:
#          headers.append(j)

# ### Provides me with headers that correspond across data (ideally would be interactive)
# counts = pd.Series(headers).value_counts()
# # noComments = count.drop('Comments')

# counts1 = counts[counts > 1]

# df1 = dfs[0]
# df2 = dfs[1]
# df3 = dfs[2]
# df4 = dfs[3]
# df5 = dfs[4]
# df6 = dfs[5]

# dfEnv = pd.merge(df1,df2, how='inner', on='Env')

# test = dfEnv.isnull().mean()
# print(test)
# noComments = counts1.drop('Comments')

# print(noComments)
# Merge Loop - this pulls out the pairs of dataframes
# df_both = []
# for i in range(len(dfs)):
#  	for j in range(len(dfs)):
#           if i == j:
#               continue
#           if i != j:
#               # print(i , j)
#               df_both.append((i,j))
         
             
# output, seen = [], set()

# for i in df_both:
#     t1 = tuple(i)
#     if t1 not in seen and tuple(reversed(i)) not in seen:
#         seen.add(t1)
#         output.append(i)
        
# print(output)

# mergedFrames = []
# #### At this point I need to access the index within noComments and loop that through out[dfs] - creates a merged DF across all time points
# for i in output:
#     # print(i)
#     for j, k in counts1.items():
#         if j in dfs[i[0]] and j in dfs[i[1]]:
            
#             test = dfs[i[0]].merge(dfs[i[1]], how='inner', on=j)
#             # print(test)
#             mergedFrames.append(test)

# df1 = mergedFrames[0]








# cTabs = df1.select_dtypes(include='number')
# df_norm = (cTabs-cTabs.min())/(cTabs.max()-cTabs.min())
# print(df_norm)

# for i in mergedFrames:
#     cTabs = i.select_dtypes(include='number')
#     df_norm = (cTabs-cTabs.min())/(cTabs.max()-cTabs.min())
#     print(df_norm)
    
# for i in mergedFrames:
#     print(i.describe())
#     headers = list(i.columns)
#     for j in range(len(headers)):
#         if headers[j].startswith('Env') and headers
#         print(headers[j])
#     columns = i.columns
    # print(i)
    
### MAKE NAN Filter for column and row
### Drop locations without a GPS location



### Nice chunck of code that imputes for each dataframe - turned into impute_pre func
 # testDtypes = common_test_DF.dtypes
# testNAs = common_test_DF.isna().sum()
# print(testDtypes.index)

# for i in range(len(testDtypes)):
#     if testDtypes[i] == 'object' and testNAs[i] != 0:
#         # print(i)
#         # print(testDtypes.index[i])
#         imputedTest = impute_categorical(common_test_DF, 'Env', testDtypes.index[i])
#         common_test_DF[testDtypes.index[i]] = imputedTest[testDtypes.index[i]]
#     if testDtypes[i] == 'int64' or testDtypes[i] == 'float64' and testNAs[i] != 0:
#         imputedTest = impute_numerical(common_test_DF, 'Env', testDtypes.index[i])
#         common_test_DF[testDtypes.index[i]] = imputedTest[testDtypes.index[i]]
            
# trainDtypes = common_train_DF.dtypes
# trainNAs = common_train_DF.isna().sum()
#     # print(testDtypes.index)
    
# for i in range(len(trainDtypes)):
#     if trainDtypes[i] == 'object' and trainNAs[i] != 0:
#         print(i)
#         print(trainDtypes.index[i])
#         imputedTest = impute_categorical(common_train_DF, 'Env', trainDtypes.index[i])
#         common_train_DF[trainDtypes.index[i]] = imputedTest[trainDtypes.index[i]]
#     if trainDtypes[i] == 'int64' or trainDtypes[i] == 'float64' and trainNAs[i] != 0:
#         imputedTest = impute_numerical(common_train_DF, 'Env', trainDtypes.index[i])
#         common_train_DF[trainDtypes.index[i]] = imputedTest[trainDtypes.index[i]]
