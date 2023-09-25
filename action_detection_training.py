import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense
from tensorflow.keras.callbacks import TensorBoard

#####
import wandb
from wandb.keras import WandbMetricsLogger


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                            ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                            ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )

def extract_keypoints(results_keypoint):

    pose = np.array([[res.x, res.y] for res in results_keypoint.pose_landmarks.landmark]).flatten() if results_keypoint.pose_landmarks else np.zeros(33*2)
    lh = np.array([[res.x, res.y] for res in results_keypoint.left_hand_landmarks.landmark]).flatten() if results_keypoint.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results_keypoint.right_hand_landmarks.landmark]).flatten() if results_keypoint.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([pose , lh, rh])

def lstm_model(modelTemp, X_train, y_train, X_val, y_val):

    configs = dict(
        num_classes = actions.shape[0],
        input_shape1 = 30,
        input_shape2 = 150,
        learning_rate = 1e-3,
        batch_size = 32,
        epochs = 150,
        architecture=  "LSTM",
    )
    model = modelTemp
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape= ( configs['input_shape1'], configs['input_shape2'])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(configs['num_classes'], activation='softmax'))

    # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')])
    
    
    # run = wandb.init(
    #     project = "AiSeminar",
    #     config = configs
    # )

    # model.fit(X_train, y_train, epochs=configs['epochs'], validation_data=(X_val, y_val),  batch_size=configs['batch_size'], 
    #             callbacks = [WandbMetricsLogger(log_freq=10)])

    # model.save('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_lstm_model.h5')
    model.load_weights('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_lstm_model.h5')

    # Close the W&B run
    # run.finish()

    return model

def gru_model(modelTemp, X_train, y_train, X_val, y_val):

    configs = dict(
        num_classes = actions.shape[0],
        input_shape1 = 30,
        input_shape2 = 150,
        learning_rate = 1e-3,
        batch_size = 32,
        epochs = 150,
        architecture=  "GRU",
    )
    model = modelTemp
    model.add(GRU(64, return_sequences=True, activation='relu', input_shape= ( configs['input_shape1'], configs['input_shape2'])))
    model.add(GRU(128, return_sequences=True, activation='relu'))
    model.add(GRU(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(configs['num_classes'], activation='softmax'))

    # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')])
    
    
    # run = wandb.init(
    #     project = "AiSeminar",
    #     config = configs
    # )

    # model.fit(X_train, y_train, epochs=configs['epochs'], validation_data=(X_val, y_val),  batch_size=configs['batch_size'], 
    #             callbacks = [WandbMetricsLogger(log_freq=10)])

    # model.save('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_gru_model.h5')
    model.load_weights('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_gru_model.h5')

    # Close the W&B run
    # run.finish()

    return model

def bigru_model(modelTemp, X_train, y_train, X_val, y_val):

    configs = dict(
        num_classes = actions.shape[0],
        input_shape1 = 30,
        input_shape2 = 150,
        learning_rate = 1e-3,
        batch_size = 32,
        epochs = 150,
        architecture=  "BIGRU",
    )
    model = modelTemp
    model.add(Bidirectional(GRU(64, return_sequences=True, activation='relu', input_shape= ( configs['input_shape1'], configs['input_shape2'])))) 
    model.add(Bidirectional(GRU(128, return_sequences=True, activation='relu'))) 
    model.add(Bidirectional(GRU(64, return_sequences=False, activation='relu'))) 
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(configs['num_classes'], activation='softmax'))

    # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')])
    
    
    # run = wandb.init(
    #     project = "AiSeminar",
    #     config = configs
    # )

    # model.fit(X_train, y_train, epochs=configs['epochs'], validation_data=(X_val, y_val),  batch_size=configs['batch_size'],  ) 
                # callbacks = [WandbMetricsLogger(log_freq=10)]

    # model.save('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_bigru_model.h5')
    model.load_weights('/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_bigru_model.h5')

    # Close the W&B run
    # run.finish()

    return model


if __name__ == '__main__':
    # Path for exported data, numpy arrays
    folder_path     = '/media/tamlab/DataStorage/SignLanguage/CodeInServer/AiSeminar/sign_data'
    DATA_PATH = os.path.join(folder_path) 

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10,  test_size=0.1)
    X_val, X_test, y_val, y_test    = train_test_split(X_test, y_test, random_state = 10,  test_size=0.5)

    print (f'{X_train.shape} ; {y_train.shape} ;  {X_val.shape} ; {y_val.shape} ; {X_test.shape} ; {y_test.shape}')


    # log_dir = os.path.join('Logs')
    # tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    # model = lstm_model(model, X_train, y_train, X_val, y_val)
    # model = gru_model(model, X_train, y_train, X_val, y_val)
    model = bigru_model(model, X_train, y_train, X_val, y_val)


    # res = model.predict(X_test)
    # print(actions[np.argmax(res[4])])

    # print (actions[np.argmax(y_test[4])]) 

    from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
    yhat = model.predict(X_test)
    print(yhat.shape)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print(yhat)
    print (accuracy_score(ytrue, yhat))

    cm = confusion_matrix(ytrue, yhat, labels = actions)
    disp = ConfusionMatrixDisplay (confusion_matrix=cm, display_labels= actions)
    disp.plot()
    plt.show()

