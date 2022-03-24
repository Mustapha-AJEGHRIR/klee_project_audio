import tensorflow as tf
import tensorflow.keras.backend as K
import os.path as p
import os

def class_mae(y_true, y_pred):
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )
    
print(tf.__version__)

models = ["CNN", "CRNN", "F-CRNN", "RNN"]
H5_PATH = p.join(
    p.dirname(p.dirname(__file__)),
    "CountNet/models/"
)
JSON_DIR = p.join(p.dirname(__file__), "json_config")
MODELS_DIR = p.join(p.dirname(__file__), "models")

# create folder if not exist
if not p.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


for model_name in models:
    with open(p.join(JSON_DIR, model_name + '_config.json')) as json_file:
        json_config = json_file.read()

    model = tf.keras.models.model_from_json(
        json_config,
        custom_objects={
            'class_mae': class_mae,
            'exp': K.exp,
        }
    )
    model.load_weights(p.join(H5_PATH, model_name + '.h5'))

    model.save(p.join(MODELS_DIR, model_name))

    print(model_name, "Saved !")
# print(model.summary())



