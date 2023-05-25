#%%
import numpy as np
import pandas as pd
import keras
from keras.layers import Input,Dense,Dropout,CategoryEncoding,Concatenate,LSTM,Conv1D,Reshape,Flatten
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import keras_tuner

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
def build_ec(hp):
    # environmental covariates model
    env_cov_in = Input(shape=(ec.shape[1],))
    env_cov_dropout = Dropout(rate=hp.Float('rate', min_value=0, max_value=0.7, step=0.1))(env_cov_in)
    env_cov_1 = Dense(units=hp.Int("units_1", min_value=8, max_value=32, step=8), activation='ReLU')(env_cov_dropout)
    env_cov_2 = Dense(units=hp.Int("units_2", min_value=2, max_value=16, step=2), activation='ReLU')(env_cov_1)
    env_cov_out = Dense(1, activation='ReLU')(env_cov_2)

    env_cov_model = keras.Model(env_cov_in,env_cov_out)
    env_cov_model.compile(optimizer='adam', loss='mean_squared_error')
    return env_cov_model

def build_loc(hp):
    # location
    loc_in = Input(shape=(loc.shape[1],))
    loc_dropout = Dropout(rate=hp.Float('rate', min_value=0, max_value=0.7, step=0.1))(loc_in)
    loc_1 = Dense(units=hp.Int("units_1", min_value=8, max_value=32, step=8), activation='ReLU')(loc_dropout)
    loc_2 = Dense(units=hp.Int("units_2", min_value=2, max_value=16, step=2), activation='ReLU')(loc_1)
    loc_out = Dense(1, activation='ReLU')(loc_2)

    loc_model = keras.Model(loc_in,loc_out)
    loc_model.compile(optimizer='adam', loss='mean_squared_error')
    return loc_model

def build_meta(hp):
    # metadata model (excl. location)
    meta_in = Input(shape=(meta.shape[1],))
    meta_dropout = Dropout(rate=hp.Float('rate', min_value=0, max_value=0.7, step=0.1))(meta_in)
    meta_1 = Dense(units=hp.Int("units_1", min_value=8, max_value=32, step=8), activation='ReLU')(meta_dropout)
    meta_2 = Dense(units=hp.Int("units_2", min_value=2, max_value=16, step=2), activation='ReLU')(meta_1)
    meta_out = Dense(1, activation='ReLU')(meta_2)

    meta_model = keras.Model(meta_in,meta_out)
    meta_model.compile(optimizer='adam', loss='mean_squared_error')
    return meta_model

def build_hybrid(hp):
    # hybrid model
    hybrid_in = Input(shape=(hybrid.shape[1],))
    hybrid_categorical = CategoryEncoding(num_tokens=max(np.unique(hybrid.values))+1,output_mode='multi_hot')(hybrid_in)
    hybrid_1 = Dense(units=hp.Int("units_1", min_value=8, max_value=32, step=8), activation='ReLU')(hybrid_categorical)
    hybrid_2 = Dense(units=hp.Int("units_2", min_value=2, max_value=16, step=2), activation='ReLU')(hybrid_1)
    hybrid_out = Dense(1, activation='ReLU')(hybrid_2)

    hybrid_model = keras.Model(hybrid_in,hybrid_out)
    hybrid_model.compile(optimizer='adam', loss='mean_squared_error')
    return hybrid_model

def build_soil(hp):
# soil model
    soil_in = Input(shape=(soil.shape[1],))
    soil_dropout = Dropout(rate=hp.Float('rate', min_value=0, max_value=0.7, step=0.1))(soil_in)
    soil_1 = Dense(units=hp.Int("units_1", min_value=8, max_value=32, step=8), activation='ReLU')(soil_dropout)
    soil_2 = Dense(units=hp.Int("units_2", min_value=2, max_value=16, step=2), activation='ReLU')(soil_1)
    soil_out =  Dense(1, activation='ReLU')(soil_2)

    soil_model = keras.Model(soil_in,soil_out)
    soil_model.compile(optimizer='adam', loss='mean_squared_error')
    return soil_model

def build_weather(hp):
# weather model
    weather_in = Input(shape=(weather.shape[1],1,))
    #weather_1 = LSTM(units=hp.Int("units_1", min_value=8, max_value=16, step=8),activation=hp.Choice('activation',['tanh','ReLU']),input_shape=(weather.shape[1],None))(weather_dropout)
    weather_1 = Conv1D(filters=hp.Int("filters_1", min_value=8, max_value=64, step=4),kernel_size=hp.Int("kernel_size_1", min_value=8, max_value=64, step=4),strides=hp.Int("strides_1", min_value=1, max_value=10, step=1),input_shape=(weather.shape[1],None))(weather_in)
    weather_flatten = Flatten()(weather_1)
    weather_dropout = Dropout(rate=hp.Float('rate1', min_value=0, max_value=0.9, step=0.1))(weather_flatten)
    weather_2 = Dense(units=hp.Int("units_2", min_value=2, max_value=32, step=2), activation='ReLU')(weather_dropout)
    weather_out =  Dense(1, activation='ReLU')(weather_2)

    weather_model = keras.Model(weather_in,weather_out)
    weather_model.compile(optimizer='adam', loss='mean_squared_error')
    return weather_model
#%%
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_weather,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=2,
    overwrite=True,
    directory="./tuning_results",
    project_name="ec_submodel",
)
tuner.search_space_summary()
# %%
all_data = weather
train_data = all_data.values[0:52660,:]
val_data = all_data.values[52660:,:]
train_y = y.values[0:52660]
val_y = y.values[52660:]
# %%
tuner.search(train_data, train_y, epochs=2, validation_data=(val_data, val_y))
tuner.results_summary()
# %%
