#%%
import numpy as np
import pandas as pd
import keras
from keras.layers import Input,Dense,Dropout,CategoryEncoding,Concatenate,LSTM,Conv1D,Reshape,Flatten
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

#%%

df = pd.read_csv('../Daniel/data/training_Meta+Hybrid+EC+Weather+Soil.csv')
df.dropna(inplace=True)

# shuffle the DataFrame rows
df = df.sample(frac = 1)

y = df['Yield_Mg_ha']
ec = df[[c for c in df.columns if 'ec_' in c]]
meta = df[[c for c in df.columns if 'meta_' in c and 'Latitude' not in c and 'Longitude' not in c]]
loc = df[[c for c in df.columns if 'meta_' in c and 'Latitude' in c or 'Longitude' in c]]
hybrid = df[[c for c in df.columns if 'hybrids_' in c]].astype(int)
soil = df[[c for c in df.columns if 'soil_' in c]]
weather = df[[c for c in df.columns if 'weather_' in c]]
#%%

# environmental covariates model
env_cov_in = Input(shape=(ec.shape[1],))
env_cov_1 = Dense(32, activation='ReLU')(env_cov_in)
env_cov_2 = Dense(16, activation='ReLU')(env_cov_1)
env_cov_out = Dense(1, activation='ReLU')(env_cov_2)

env_cov_model = keras.Model(env_cov_in,env_cov_out)

# location
loc_in = Input(shape=(loc.shape[1],))
loc_dropout = Dropout(rate=0.0)(loc_in)
loc_1 = Dense(8,activation='linear')(loc_dropout)
loc_2 = Dense(12,activation='linear')(loc_1)
loc_out = Dense(1, activation='ReLU')(loc_2)

loc_model = keras.Model(loc_in,loc_out)

# metadata model (excl. location)
meta_in = Input(shape=(meta.shape[1],))
meta_dropout = Dropout(rate=0.1)(meta_in)
meta_1 = Dense(32,activation='ReLU')(meta_dropout)
meta_2 = Dense(2,activation='ReLU')(meta_1)
meta_out = Dense(1, activation='ReLU')(meta_2)

meta_model = keras.Model(meta_in,meta_out)

# hybrid model
hybrid_in = Input(shape=(hybrid.shape[1],))
hybrid_categorical = CategoryEncoding(num_tokens=max(np.unique(hybrid.values))+1,output_mode='multi_hot')(hybrid_in)
hybrid_1 = Dense(32,activation='ReLU')(hybrid_categorical)
hybrid_2 = Dense(14,activation='ReLU')(hybrid_1)
hybrid_out = Dense(1, activation='ReLU')(hybrid_2)

hybrid_model = keras.Model(hybrid_in,hybrid_out)

# soil model
soil_in = Input(shape=(soil.shape[1],))
soil_dropout = Dropout(rate=0.4)(soil_in)
soil_1 = Dense(24,activation='ReLU')(soil_dropout)
soil_2 = Dense(8,activation='ReLU')(soil_1)
soil_out =  Dense(1, activation='ReLU')(soil_2)

soil_model = keras.Model(soil_in,soil_out)

# weather model
weather_in = Input(shape=(weather.shape[1],1,))
weather_1 = Conv1D(filters=64,kernel_size=28,strides=1,input_shape=(weather.shape[1],None))(weather_in)
weather_flatten = Flatten()(weather_1)
weather_dropout = Dropout(rate=0.6)(weather_flatten)
weather_2 = Dense(32)(weather_dropout)
weather_out =  Dense(1, activation='ReLU')(weather_2)

weather_model = keras.Model(weather_in,weather_out)

# fusion model
fusion_in = Concatenate()([env_cov_2,loc_2,meta_2,hybrid_2,soil_2])
fusion_d1 = Dropout(rate=0.5)(fusion_in)
fusion_1 = Dense(1, activation='ReLU')(fusion_d1)
fusion_2 = Dense(64, activation='ReLU')(fusion_1)
fusion_d3 = Dropout(rate=0.1)(fusion_2)
fusion_out = Dense(1)(fusion_d3)

fusion_model = keras.Model([env_cov_in,loc_in,meta_in,hybrid_in,soil_in,weather_in],fusion_out)

#%% fusion model training
test_year = 2021

ec_train = ec.values[~df['Env'].str.contains('2021'),:]
ec_test = ec.values[df['Env'].str.contains('2021'),:]

loc_train = loc.values[~df['Env'].str.contains('2021'),:]
loc_test = loc.values[df['Env'].str.contains('2021'),:]

meta_train = meta.values[~df['Env'].str.contains('2021'),:]
meta_test = meta.values[df['Env'].str.contains('2021'),:]

hybrid_train = hybrid.values[~df['Env'].str.contains('2021'),:]
hybrid_test = hybrid.values[df['Env'].str.contains('2021'),:]

soil_train = soil.values[~df['Env'].str.contains('2021'),:]
soil_test = soil.values[df['Env'].str.contains('2021'),:]

weather_train = weather.values[~df['Env'].str.contains('2021'),:]
weather_test = weather.values[df['Env'].str.contains('2021'),:]

y_train = y.values[~df['Env'].str.contains('2021')]
y_test = y.values[df['Env'].str.contains('2021')]

fusion_model.compile(optimizer='adam', loss='mean_squared_error')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
fusion_model.fit([ec_train,loc_train,meta_train,hybrid_train,soil_train,weather_train],y_train,epochs=1000,batch_size=32,shuffle=True,validation_data=([ec_test,loc_test,meta_test,hybrid_test,soil_test,weather_test],y_test),callbacks=[es])
#%%
fusion_model.evaluate([ec_test,loc_test,meta_test,hybrid_test,soil_test,weather_test],y_test)

# %%
df = pd.read_csv('../Daniel/data/test_Meta+Hybrid+EC+Weather+Soil.csv')

ec = df[[c for c in df.columns if 'ec_' in c]]
meta = df[[c for c in df.columns if 'meta_' in c and 'Latitude' not in c and 'Longitude' not in c and 'Week' not in c]]
loc = df[[c for c in df.columns if 'meta_' in c and 'Latitude' in c or 'Longitude' in c]]
hybrid = df[[c for c in df.columns if 'hybrids_' in c]].astype(int)
soil = df[[c for c in df.columns if 'soil_' in c]]
weather = df[[c for c in df.columns if 'weather_' in c]]
# %%
y = fusion_model.predict([ec,meta,loc,hybrid,soil,weather])
# %%
