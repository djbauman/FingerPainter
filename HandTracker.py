import time
import cv2
import mediapipe as mp

class Tracker():
  def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.85, tracking_confidence=0.85):
    self.mode = mode
    self.max_hands = max_hands
    self.model_complexity = model_complexity
    self.detection_confidence = detection_confidence
    self.tracking_confidence = tracking_confidence
    self.mp_hands = mp.solutions.hands
    self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_confidence, self.tracking_confidence)
    self.mp_draw = mp.solutions.drawing_utils
    self.tip_ids = [4,8,12,16,20]                                        # Finger tip IDs


  def track_hands(self, img, draw=True):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       # Convert the BGR image to RGB
    self.results = self.hands.process(img_RGB)
    if self.results.multi_hand_landmarks:
      for hand_landmarks in self.results.multi_hand_landmarks:
        if draw:
          self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    return img


  def get_landmarks(self, img, hand_num=0, draw=True, log=True):                                       
    self.landmarks = []
    if self.results.multi_hand_landmarks:
      hand = self.results.multi_hand_landmarks[hand_num]
      for id, lm in enumerate(hand.landmark):
        h, w, c = img.shape                                                       # h = 720, w = 1280, c = 3
        x_pixel, y_pixel = int(lm.x*w), int(lm.y*h)                               # Convert landmark locations from decimal ratios to pixel values
        self.landmarks.append([id, x_pixel, y_pixel])
        if log:
          print(f"ID: {id} X: {x_pixel} Y: {y_pixel}")                            # (Logging) Print landmarks and their locations to the console
        if draw:                                                                  # (Example) Taking action based on a specific landmark.
          cv2.circle(img, (x_pixel, y_pixel), 10, (255,225,0), cv2.FILLED)
    return self.landmarks


  def read_fingers(self):
    fingers = []
    # Thumb
    if self.landmarks[self.tip_ids[0]][1] > self.landmarks[self.tip_ids[0] - 1][1]: # TODO: refine this calculation to work for either thumb
      fingers.append(1)
    else:
      fingers.append(0)
    
    # Other fingers
    for id in range(1,5):
      if self.landmarks[self.tip_ids[id]][2] < self.landmarks[self.tip_ids[id] - 2][2]:
        fingers.append(1)
      else:
        fingers.append(0)    
    return fingers


# Finger tracking demo
def main():
  prev_time = 0
  current_time = 0
  cap = cv2.VideoCapture(0)
  cap.set(3, 1280)
  cap.set(4, 720)
  detector = Tracker()
  fingers = []

  while True:
    result, img = cap.read()
    img = detector.track_hands(img)
    landmarks = detector.get_landmarks(img, log=True)
    if len(landmarks) != 0:
      fingers = detector.read_fingers()

    # Calculate framerate
    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    # Display window
    cv2.putText(img, "FPS: "+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,215,0), 2)
    cv2.putText(img, str(fingers), (450,100), cv2.FONT_HERSHEY_COMPLEX, 3, (200,20,200), 2)
    cv2.imshow("HandTracker 🖐", img)
    cv2.waitKey(1)


if __name__ == "__main__":
  main()