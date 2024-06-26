import cv2
import numpy as np
import os
import mediapipe as mp
import onnx
import onnxruntime as rt

import openai

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

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
    return response.choices[0].message["content"]

if __name__ == '__main__':

    openai.api_key = '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'

    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('/home/tamlab/Downloads/sign-language-from-pc') 

    # Actions that we try to detect
    actions = np.array(['hello', 'thanks', 'iloveyou', 'book' , 'standby' , 'need' , 'I', 'help'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30

    # Folder start
    start_folder = 30




    colors = [(245,117,16), (117,245,16), (16,117,245), 
        (180,100,16), (100,180,16), (16,100,180),  
        (100,17,160), (17,100,160)]



    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
	    # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
		
            # Draw landmarks
            draw_styled_landmarks(image, results)
		
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
		
            if len(sequence) == 30:
	        # providers = ['CPUExecutionProvider']
                m = rt.InferenceSession('/home/tamlab/Downloads/sign-language-from-pc/sign_model_7.onnx')
                onnx_pred = m.run(None, {"input" : np.expand_dims(sequence, axis = 0).astype(np.float32)})
                res = np.asarray(onnx_pred[0][0])
                # print([res.shape])
                predictions.append(np.argmax(res))
	 
		    
	        #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    temp = ''
                    if res[np.argmax(res)] > threshold: 
	            
                       if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])
		            
                            temp = action[np.argmax(res)]
                      
                       # if (actions[np.argmax(res)]  != 'standby') and (actions[np.argmax(res)] != temp):
                       #		prompt = actions[np.argmax(res)]
                       #		response = get_completion(prompt)
                       #		print(f'{prompt=} >> {response}')


                if len(sentence) > 5: 
                    sentence = sentence[-5:]

    		# Viz probabilities
                image = prob_viz(res, actions, image, colors)
                
            # cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            # cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		
            # Show to screen
            cv2.imshow('AiSeminar', image)

    	    # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
