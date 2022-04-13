# %% [markdown]
# # Imports

# %%
import tensorflow as tf

import onnxruntime as ort
import numpy as np
from time import time

# %% [markdown]
# Run tests only on cpu

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # no tf logging

# %% [markdown]
# # Speed test

# %%
RUNS = 100

# %% [markdown]
# ## Normal model

# # %%
# model = tf.keras.models.load_model("saved/tf_model")
# input = tf.random.normal([1, 201, 498])

# tik = time()
# for _ in range(RUNS):
#     model(input)
# tok = time()
# print(model(input))
# print("Duration :{:.2f}ms".format(((tok-tik)/RUNS)*1000))

# %% [markdown]
# ## ONNX

# %%
print("**********")
print("ONNX".center(10, "-"))
print("**********")
tf_random_tensor = np.random.rand(1, 201, 498)
tf_random_tensor = tf_random_tensor.astype(np.float32)

ort_sess = ort.InferenceSession('onnx/f-crnn.onnx')
outputs = ort_sess.run([], {'x': tf_random_tensor})
# Print Result
# result = outputs[0].argmax(axis=1)
print("\tActivations :", outputs[0][0])
# print("label :",result[0])

tik = time()
for _ in range(RUNS):
    ort_sess.run([], {'x': tf_random_tensor})
tok = time()
print("\tDuration : {:.2f}ms".format(((tok-tik)/RUNS)*1000))

# %% [markdown]
# ## TF-lite

# %%
print("**********")
print("TF-lite".center(10, "-"))
print("**********")
converter = tf.lite.TFLiteConverter.from_saved_model("saved/tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# %%
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, tf.random.normal([1, 201, 498]))

tik = time()
for _ in range(RUNS):
    interpreter.invoke()
tok = time()
output = interpreter.tensor(output_index)()[0]
print("\tActivations :", output)
print("\tDuration : {:.2f}ms".format(((tok-tik)/RUNS)*1000))


