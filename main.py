#import needed modules
import tensorflow as tf
import pandas as pd
import numpy as np
df = pd.read_csv("diabetes.csv")
#I put this line as merely comment because I'm not sure only keeping 3 features which aren't super accurate on their own is a good ideaI
#In addition feature engineering might be performed later on
#df = df.drop(columns=["Pregnancies","BloodPressure","SkinThickness","Insulin","DiabetesPedigreeFunction"])
#randomise
df = df.reindex(np.random.permutation(df.index))
train_set, test_set = df.iloc[:600],df.iloc[600:]
#build and compile the model
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(8,)),
tf.keras.layers.Dense(128,activation="relu"),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(64,activation="relu"),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1,activation="sigmoid"),
])
model.compile(
optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"])
y_train = train_set["Outcome"]
y_test = test_set["Outcome"]
x_train = train_set.drop(columns=["Outcome"])
x_test = test_set.drop(columns=["Outcome"])
model.fit(x=x_train,y=y_train,validation_split=0.25,epochs=300)
model.evaluate(x_test,y_test)
