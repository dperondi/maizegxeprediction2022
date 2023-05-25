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
    env_cov_dropout = Dropout(rate=hp.Float('ec_rate', min_value=0, max_value=0.5, step=0.1))(env_cov_in)
    env_cov_1 = Dense(units=hp.Int("ec_units_1", min_value=8, max_value=32, step=8), activation='ReLU')(env_cov_dropout)
    env_cov_2 = Dense(units=hp.Int("ec_units_2", min_value=2, max_value=16, step=2), activation='ReLU')(env_cov_1)

    # location
    loc_in = Input(shape=(loc.shape[1],))
    loc_dropout = Dropout(rate=hp.Float('loc_rate', min_value=0, max_value=0.5, step=0.1))(loc_in)
    loc_1 = Dense(units=hp.Int("loc_units_1", min_value=8, max_value=32, step=8), activation='ReLU')(loc_dropout)
    loc_2 = Dense(units=hp.Int("loc_units_2", min_value=2, max_value=16, step=2), activation='ReLU')(loc_1)

    # metadata model
    meta_in = Input(shape=(meta.shape[1],))
    meta_dropout = Dropout(rate=hp.Float('meta_rate', min_value=0, max_value=0.5, step=0.1))(meta_in)
    meta_1 = Dense(units=hp.Int("meta_units_1", min_value=8, max_value=32, step=8), activation='ReLU')(meta_dropout)
    meta_2 = Dense(units=hp.Int("meta_units_2", min_value=2, max_value=16, step=2), activation='ReLU')(meta_1)

    hybrid_in = Input(shape=(hybrid.shape[1],))
    hybrid_categorical = CategoryEncoding(num_tokens=max(np.unique(hybrid.values))+1,output_mode='multi_hot')(hybrid_in)
    hybrid_1 = Dense(units=hp.Int("hybrid_units_1", min_value=8, max_value=32, step=8), activation='ReLU')(hybrid_categorical)
    hybrid_2 = Dense(units=hp.Int("hybrid_units_2", min_value=2, max_value=16, step=2), activation='ReLU')(hybrid_1)

    # soil model
    soil_in = Input(shape=(soil.shape[1],))
    soil_dropout = Dropout(rate=hp.Float('soil_rate', min_value=0, max_value=0.5, step=0.1))(soil_in)
    soil_1 = Dense(units=hp.Int("soil_units_1", min_value=8, max_value=32, step=8), activation='ReLU')(soil_dropout)
    soil_2 = Dense(units=hp.Int("soil_units_2", min_value=2, max_value=16, step=2), activation='ReLU')(soil_1)


    # weather model
    weather_in = Input(shape=(weather.shape[1],1,))
    #weather_1 = LSTM(units=hp.Int("units_1", min_value=8, max_value=16, step=8),activation=hp.Choice('activation',['tanh','ReLU']),input_shape=(weather.shape[1],None))(weather_dropout)
    weather_1 = Conv1D(filters=hp.Int("weather_filters", min_value=8, max_value=64, step=4),kernel_size=hp.Int("weather_kernel_size", min_value=8, max_value=64, step=4),strides=hp.Int("weather_strides", min_value=1, max_value=10, step=1),input_shape=(weather.shape[1],None))(weather_in)
    weather_flatten = Flatten()(weather_1)
    weather_dropout = Dropout(rate=hp.Float('weather_rate', min_value=0, max_value=0.9, step=0.1))(weather_flatten)
    weather_2 = Dense(units=hp.Int("weather_units_2", min_value=2, max_value=32, step=2), activation='ReLU')(weather_dropout)

    # fusion model
    fusion_in = Concatenate()([env_cov_2,loc_2,meta_2,hybrid_2,soil_2,weather_2])
    fusion_d1 = Dropout(rate=hp.Float('fusion_rate_1', min_value=0, max_value=0.5, step=0.1))(fusion_in)
    fusion_1 = Dense(units=hp.Int("fusion_units_1", min_value=4, max_value=64, step=4), activation='ReLU')(fusion_d1)
    fusion_d2 = Dropout(rate=hp.Float('fusion_rate_2', min_value=0, max_value=0.5, step=0.1))(fusion_1)
    fusion_2 = Dense(units=hp.Int("fusion_units_2", min_value=4, max_value=64, step=4), activation='ReLU')(fusion_d2)
    fusion_d3 = Dropout(rate=hp.Float('fusion_rate_3', min_value=0, max_value=0.5, step=0.1))(fusion_2)
    fusion_out = Dense(1)(fusion_d3)

    fusion_model = keras.Model([env_cov_in,loc_in,meta_in,hybrid_in,soil_in,weather_in],fusion_out)
    fusion_model.compile(optimizer='adam', loss='mean_squared_error')
    return fusion_model

#%% hyperparameter tuning
failed_parameters = []
class CVTuner(keras_tuner.engine.tuner.Tuner):
  def run_trial(self, trial, x, y, batch_size=32, epochs=1):
    val_losses = []
    for year in ['2021','2020','2019','2018','2017','2016']:
        train_indices = ~df['Env'].str.contains(year)
        test_indices = df['Env'].str.contains(year)
        x_train = []
        x_test = []
        for subset in x:
            x_train.append(subset[train_indices])
            x_test.append(subset[test_indices])
        y_train, y_test = y[train_indices], y[test_indices]
        try:
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=False)
            val_losses.append(model.evaluate(x_test, y_test))
        except:
            val_losses.append(1000)
            print('parameters failed')
            failed_parameters.append(trial.hyperparameters)
    self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
    #self.save_model(trial.trial_id, model)
tuner = CVTuner(
  hypermodel=build_fusion,
  oracle=keras_tuner.oracles.BayesianOptimization(
    objective='val_loss',
    num_initial_points=48,
    max_trials=100))

'''
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_fusion,
    objective="val_loss",
    max_trials=100,
    executions_per_trial=1,
    overwrite=True,
    directory="./tuning_results",
    project_name="ec_submodel",
)
'''
tuner.search_space_summary()
# %%
'''
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
'''
# %%
tuner.search([ec,loc,meta,hybrid,soil,weather], y, epochs=2)
tuner.results_summary()
# %%
