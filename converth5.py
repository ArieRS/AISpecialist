import tf2onnx
import onnxruntime as rt
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou', 'book', 'standby', 'need', 'I', 'help'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

label_map = {label:num for num, label in enumerate(actions)}
# print (label_map)
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,150)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_lstm_model.h5')



spec = (tf.TensorSpec((None, 30, 150), tf.float32, name="input"),)
# output_path = os.path.join('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar', 'sign_model.onnx')
output_path = 'sign_model_7' + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

#############
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(output_names, {"input": X_test.astype(np.float32) })

# onnx_pred = np.argmax(onnx_pred[0], axis=1).tolist()

print('ONNX Predicted:', np.asarray(onnx_pred))

# make sure ONNX and keras have the same results
# np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-5)