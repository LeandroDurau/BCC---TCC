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

target = df[columns[0],'Y']
dataset = tf.data.Dataset.from_tensor_slices((df[columns[0],'Y'].values, target.values))

train_dataset = dataset.shuffle(len(df)).batch(1)

def get_compiled_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(16, return_sequences=True),    
    tf.keras.layers.SimpleRNN(8),
    tf.keras.layers.Dense(1)
    ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)