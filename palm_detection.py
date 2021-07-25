import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1) # for webcam
mphands = mp.solutions.hands
hands = mphands.Hands() 
mpdraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    
    _ , frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(frame, handlms, mphands.HAND_CONNECTIONS)
            for idx,lm in enumerate(handlms.landmark):
                
                H,W,_ = frame.shape
                centerX,centerY = int(lm.x*W), int(lm.y*H)
                
                if idx == 0:
                    cv2.circle(frame, (centerX,centerY), 5, (255,0,255), cv2.FILLED)
                    
    
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    
    cv2.putText(frame, "frame rate: "+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.imshow('Hand Palm Detection', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
