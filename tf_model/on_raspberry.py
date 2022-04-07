import sounddevice as sd    
import onnxruntime as ort
import numpy as np
import scipy.signal
from scipy.io.wavfile import write
from time import time
import librosa


# ---------------------------------------------------------------------------- #
#                                 Configuration                                #
# ---------------------------------------------------------------------------- #

FFT_N_PERSEG = 400
FFT_N_OVERLAP = 240
FFT_WINDOW_TYPE = "tukey"
EPS = 1e-8
SAMPLE_RATE = 16000 # sample rate
SECONDS = 5  # Duration of recording

# ---------------------------------------------------------------------------- #
#                                    Models                                    #
# ---------------------------------------------------------------------------- #

gender_count_sess = ort.InferenceSession('onnx/f-crnn.onnx')
people_counter_sess = ort.InferenceSession('../old_model_conversion/onnx/CRNN_ONNX.onnx')
input_names = {
    "p" : "zero1_input",
    "g" : "x",
}

ort_sess = people_counter_sess
model = "p"

while True :
    # ---------------------------------------------------------------------------- #
    #                                 Record Audio                                 #
    # ---------------------------------------------------------------------------- #
    print("Recording ...")
    tik = time()
    myrecording = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    myrecording = myrecording[:,0]
    tok = time()
    print("Recording done in %.2f seconds" % (tok - tik))



    write('tmp/audio_recorded.wav', SAMPLE_RATE, myrecording)  # Save as WAV file 

    # ---------------------------------------------------------------------------- #
    #                               Preprocess Audio                               #
    # ---------------------------------------------------------------------------- #
    tik = time()
    if model == "g":
        _, _, fft_rec = scipy.signal.spectrogram(myrecording,
                                                fs = SAMPLE_RATE,
                                                nperseg = FFT_N_PERSEG,
                                                noverlap = FFT_N_OVERLAP,
                                                window = FFT_WINDOW_TYPE
                                                )

        fft_rec = fft_rec / (np.linalg.norm(fft_rec, axis=0, keepdims=True) + EPS)
        fft_rec = fft_rec.astype(np.float32)
        fft_rec = fft_rec[None, :, :] # Add the batch dimension = 1
    elif model == "p":
        fft_rec = np.abs(librosa.stft(myrecording, n_fft=400, hop_length=160)).T
        fft_rec = fft_rec[:500, :]
        fft_rec = fft_rec / (np.linalg.norm(fft_rec, axis=0, keepdims=True) + EPS)
        fft_rec = fft_rec.astype(np.float32)
        fft_rec = fft_rec[None, None, :, :] # Add the batch dimension = 1
        
    tok = time()
    print("pre processing time : {:.3f}s".format(tok-tik))



    # ---------------------------------------------------------------------------- #
    #                                   Inference                                  #
    # ---------------------------------------------------------------------------- #
    tik = time()
    input_name = input_names[model]
    outputs = ort_sess.run([], {input_name: fft_rec})
    tok = time()
    if model == "p":
        print("\t=> Activations :   ", outputs[0][0])
        print("\t=> Total amount of people :   ", np.argmax(outputs[0][0]))
        print("\t=> Duration :   {:.3f}s".format(tok-tik))
    else:
        print("\t=> Activations [M, F] :   ", outputs[0][0])
        print("\t=> Total amount of people :   ", round(outputs[0][0].sum()))
        print("\t=> Duration :   {:.3f}s".format(tok-tik))


    
    # ---------------------------------------------------------------------------- #
    #                                     Menu                                     #
    # ---------------------------------------------------------------------------- #
    print("****")
    print("q - Quit")
    print("g - Gender counter")
    print("p - People counter")
    
    ans = input("Enter : ")
    if ans == "q":
        break
    elif ans == "g":
        ort_sess = gender_count_sess
        model = ans
    elif ans == "p":
        model = ans
        ort_sess = people_counter_sess
    else :
        print("Continuing with the same model ...")
    
    