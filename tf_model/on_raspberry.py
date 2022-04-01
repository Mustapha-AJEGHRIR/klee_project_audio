import sounddevice as sd    
import onnxruntime as ort
import numpy as np
import scipy.signal
from time import time


# ---------------------------------------------------------------------------- #
#                                 Configuration                                #
# ---------------------------------------------------------------------------- #

FFT_N_PERSEG = 400
FFT_N_OVERLAP = 240
FFT_WINDOW_TYPE = "tukey"
EPS = 1e-8
SAMPLE_RATE = 16000

ort_sess = ort.InferenceSession('onnx/f-crnn.onnx')


# ---------------------------------------------------------------------------- #
#                                 Record Audio                                 #
# ---------------------------------------------------------------------------- #
fs = SAMPLE_RATE  # Sample rate
seconds = 5  # Duration of recording
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
myrecording = myrecording[:,0]


# ---------------------------------------------------------------------------- #
#                               Preprocess Audio                               #
# ---------------------------------------------------------------------------- #
_, _, fft_rec = scipy.signal.spectrogram(myrecording,
                                        fs = SAMPLE_RATE,
                                        nperseg = FFT_N_PERSEG,
                                        noverlap = FFT_N_OVERLAP,
                                        window = FFT_WINDOW_TYPE
                                        )

fft_rec = fft_rec / (np.linalg.norm(fft_rec, axis=0, keepdims=True) + EPS)
fft_rec = fft_rec.astype(np.float32)
fft_rec = fft_rec[None, :, :] # Add the batch dimension = 1
fft_rec.shape



# ---------------------------------------------------------------------------- #
#                                   Inference                                  #
# ---------------------------------------------------------------------------- #
tik = time()
outputs = ort_sess.run([], {'x': fft_rec})
tok = time()
print("Activations [M, F] :", outputs[0][0])
print("Total amount of people :", round(outputs[0][0].sum()))
print("Duration :{:.3f}s".format(tok-tik))