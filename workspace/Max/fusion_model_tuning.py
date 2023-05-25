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
def build_fusion(hp):
    # environmental covariates model
    env_cov_in = Input(shape=(ec.shape[1],))
    env_cov_dropout = Dropout(rate=0.5)(env_cov_in)
    env_cov_1 = Dense(8, activation='ReLU')(env_cov_dropout)
    env_cov_2 = Dense(12, activation='ReLU')(env_cov_1)
    env_cov_out = Dense(1, activation='ReLU')(env_cov_2)

    env_cov_model = keras.Model(env_cov_in,env_cov_out)

    # location
    loc_in = Input(shape=(loc.shape[1],))
    loc_dropout = Dropout(rate=0.1)(loc_in)
    loc_1 = Dense(24,activation='linear')(loc_dropout)
    loc_2 = Dense(8,activation='linear')(loc_1)
    loc_out = Dense(1, activation='ReLU')(loc_2)

    loc_model = keras.Model(loc_in,loc_out)

    # metadata model (excl. location)
    meta_in = Input(shape=(meta.shape[1],))
    meta_dropout = Dropout(rate=0.4)(meta_in)
    meta_1 = Dense(16,activation='ReLU')(meta_dropout)
    meta_2 = Dense(12,activation='ReLU')(meta_1)
    meta_out = Dense(1, activation='ReLU')(meta_2)

    meta_model = keras.Model(meta_in,meta_out)

    # hybrid model
    hybrid_in = Input(shape=(hybrid.shape[1],))
    hybrid_categorical = CategoryEncoding(num_tokens=max(np.unique(hybrid.values))+1,output_mode='multi_hot')(hybrid_in)
    hybrid_1 = Dense(8,activation='ReLU')(hybrid_categorical)
    hybrid_2 = Dense(12,activation='ReLU')(hybrid_1)
    hybrid_out = Dense(1, activation='ReLU')(hybrid_2)

    hybrid_model = keras.Model(hybrid_in,hybrid_out)

    # soil model
    soil_in = Input(shape=(soil.shape[1],))
    soil_dropout = Dropout(rate=0.2)(soil_in)
    soil_1 = Dense(24,activation='ReLU')(soil_dropout)
    soil_2 = Dense(4,activation='ReLU')(soil_1)
    soil_out =  Dense(1, activation='ReLU')(soil_2)

    soil_model = keras.Model(soil_in,soil_out)

    # weather model
    weather_in = Input(shape=(weather.shape[1],1,))
    weather_1 = Conv1D(filters=64,kernel_size=8,strides=10,input_shape=(weather.shape[1],None))(weather_in)
    weather_flatten = Flatten()(weather_1)
    weather_dropout = Dropout(rate=0.6)(weather_flatten)
    weather_2 = Dense(10)(weather_dropout)
    weather_out =  Dense(1, activation='ReLU')(weather_2)

    weather_model = keras.Model(weather_in,weather_out)

    # fusion model
    fusion_in = Concatenate()([env_cov_2,loc_2,meta_2,hybrid_2,soil_2,weather_2])
    fusion_d1 = Dropout(rate=hp.Float('rate1', min_value=0.1, max_value=0.5, step=0.1))(fusion_in)
    fusion_1 = Dense(units=hp.Int("units_1", min_value=4, max_value=64, step=4), activation='ReLU')(fusion_d1)
    fusion_d2 = Dropout(rate=hp.Float('rate2', min_value=0.1, max_value=0.5, step=0.1))(fusion_1)
    fusion_2 = Dense(units=hp.Int("units_2", min_value=4, max_value=64, step=4), activation='ReLU')(fusion_d2)
    fusion_d3 = Dropout(rate=hp.Float('rate3', min_value=0.1, max_value=0.5, step=0.1))(fusion_2)
    fusion_out = Dense(1)(fusion_d3)

    fusion_model = keras.Model([env_cov_in,loc_in,meta_in,hybrid_in,soil_in,weather_in],fusion_out)
    fusion_model.compile(optimizer='adam', loss='mean_squared_error')
    return fusion_model

#%% hyperparameter tuning

tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_fusion,
    objective="val_loss",
    max_trials=50,
    executions_per_trial=1,
    overwrite=True,
    directory="./tuning_results",
    project_name="ec_submodel",
)
tuner.search_space_summary()
# %%
train_ec = ec.values[0:47000,:]
val_ec = ec.values[47000:,:]

train_loc = loc.values[0:47000,:]
val_loc = loc.values[47000:,:]

train_meta = meta.values[0:47000,:]
val_meta = meta.values[47000:,:]

train_hybrid = hybrid.values[0:47000,:]
val_hybrid = hybrid.values[47000:,:]

train_soil = soil.values[0:47000,:]
val_soil = soil.values[47000:,:]

train_weather = weather.values[0:47000,:]
val_weather = weather.values[47000:,:]

train_y = y.values[0:47000]
val_y = y.values[47000:]
# %%
tuner.search([train_ec,train_loc,train_meta,train_hybrid,train_soil,train_weather], train_y, epochs=20, validation_data=([val_ec,val_loc,val_meta,val_hybrid,val_soil,val_weather], val_y))
tuner.results_summary()
# %%
