import onnxruntime as ort
import numpy as np
from time import time

tf_random_tensor = np.random.rand(1, 1, 500, 201)
tf_random_tensor = tf_random_tensor.astype(np.float32)
# tf_random_tensor = tf.random.uniform((1,1, 500, 201))

ort_sess = ort.InferenceSession('onnx/CRNN_ONNX.onnx')
outputs = ort_sess.run([], {'zero1_input': tf_random_tensor})
# Print Result
result = outputs[0].argmax(axis=1)
print("Activations :", outputs[0][0])
print("label :",result[0])

tik = time()
ort_sess.run([], {'zero1_input': tf_random_tensor})
tok = time()
print("Duration :{:.2f}s".format(tok-tik))