import cv2
import mediapipe as mp

def finger_direction(camera_id=0):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(camera_id)
    
    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the image to RGB, flip it horizontally for a later selfie-view mirror effect
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process the image and get hand landmarks
            results = hands.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            direction = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if len(results.multi_hand_landmarks) == 2:
                    left_hand = results.multi_hand_landmarks[0]
                    right_hand = results.multi_hand_landmarks[1]
                    left_wrist = left_hand.landmark[0]
                    right_wrist = right_hand.landmark[0]
                    
                    # Calculate normalized differences
                    diff_x = right_wrist.x - left_wrist.x
                    diff_y = right_wrist.y - left_wrist.y
                    
                    if abs(diff_y) < 0.05:  # Nearly horizontal line
                        direction = "R" if diff_x > 0 else "L"
                    else:  # Significant vertical difference
                        direction = "D" if diff_y > 0 else "U"
                        
                elif len(results.multi_hand_landmarks) == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    wrist = hand_landmarks.landmark[0]
                    index_tip = hand_landmarks.landmark[8]
                    
                    # Calculate normalized differences
                    diff_x = index_tip.x - wrist.x
                    diff_y = index_tip.y - wrist.y
                    
                    if abs(diff_x) > abs(diff_y):  # Horizontal movement
                        direction = "R" if diff_x > 0 else "L"
                    else:  # Vertical movement
                        direction = "D" if diff_y > 0 else "U"

            # Display the direction detected
            if direction:
                print(f"Direction: {direction}")

            # Show the image with hand landmarks
            cv2.imshow('MediaPipe Hands', image)
            
            # Break the loop if 'Esc' key is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    finger_direction()