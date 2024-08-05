TF_ENABLE_ONEDNN_OPTS=0
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os



#katia - 12
#nelson 16
#Amaro - 18
#Adilson - 19
#Jair - 22
#Marilene - 25
#Izabel 27
#Josui - 30
#Carlos - 32
#Arnaldo - 34

df = pd.read_csv("./dados/nelson_testeSL.csv", skiprows=[0,1,2,4,5], header=[0,1])
df = df.rename(columns={'Unnamed: 0_level_0': ''})

columns = [
        'Skeleton 016:LFTC', #0
        'Skeleton 016:RIPS', #1
        'Skeleton 016:Head', #2
        'Skeleton 016:LShoulder', #3
        'Skeleton 016:LUArm', #4
        'Skeleton 016:LFLE', #5
        ]

selectedColumn = columns[0]
selectedColumn2 = columns[5]

# outra base
'''
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
data = df['T (degC)']

data.index = df['Date Time']
#outra base fim
'''
df['Fadiga'] = ['Sim' if x > ((4/5) * len(df)) else 'Nao' for x in df['','Frame']]
df.columns = [f'{i} {j}' for i, j in df.columns]
df2 = df[['Name Time (Seconds)','Skeleton 016:LFTC X','Skeleton 016:LFTC Y','Skeleton 016:LFTC Z','Fadiga ']]


df2['Skeleton 016:LFTC X'].fillna(df['Skeleton 016:LFTC X'].mean(), inplace=True)
data = df2['Skeleton 016:LFTC X']
data.index = df2['Name Time (Seconds)']

TRAIN_SPLIT = 4000 #3 mil para treino

tf.random.set_seed(100)

uni_data = data.values

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    #if i < 1000: print(indices)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

univariate_past_history = 50  #50 observacoes anteriores
future = univariate_future_target = 5  # prever 5 observações a frente 

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'gX', 'ro']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
      future = delta
    else:
      future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

BATCH_SIZE = 256
BUFFER_SIZE = 1000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
print(x_train_uni.shape)

print(x_train_uni.shape[1])
print(x_train_uni.shape[2])
simple_rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(16, return_sequences=True, input_shape=(x_train_uni.shape[1], x_train_uni.shape[2])),    
    tf.keras.layers.SimpleRNN(8),
    tf.keras.layers.Dense(1)
])


simple_rnn_model.compile(optimizer='adam', loss='mae')

simple_rnn_model.summary()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(16, return_sequences=True, input_shape=(x_train_uni.shape[1], x_train_uni.shape[2])),
    tf.keras.layers.LSTM(8),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')


simple_lstm_model.summary()

EVALUATION_INTERVAL = 100
EPOCHS = 10

from tensorflow.keras.callbacks import EarlyStopping

#Se não houver melhoria apos 3 epocas, para o treinamento
es=EarlyStopping(monitor='val_loss', verbose=1, patience=3)

rnn_log = simple_rnn_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50, callbacks=[es])

lstm_log = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50, callbacks=[es])

simple_lstm_model.save('model.keras')

jsmodel=simple_lstm_model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(jsmodel)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

plot_train_history(rnn_log,'RNN Training and validation loss')
plot_train_history(lstm_log,'LSTM Training and validation loss')

def plot_preds(plot_data, delta=0):
    labels = ['History', 'True Future', 'RNN Prediction','LSTM Prediction']
    marker = ['.-', 'gX', 'ro' , 'bo']
    time_steps = create_time_steps(plot_data[0].shape[0])
    

    future = delta

    plt.title('Predictions')
    for i, x in enumerate(plot_data):
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

for x, y in val_univariate.take(1):
  for sample in range(10):
    plot = plot_preds([x[sample].numpy(), y[sample].numpy(),
                    simple_rnn_model.predict(x)[sample], simple_lstm_model.predict(x)[sample]], future)
    plot.show()

print(simple_rnn_model.predict(x)[0])
print(val_univariate.take(1))




