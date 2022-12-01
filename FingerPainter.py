import os
import numpy
import cv2
import HandTracker

# Collect overlay images
folderPath = "overlays"
directory = os.listdir(folderPath)
overlays = []
for path in directory:
  image = cv2.imread(f'{folderPath}/{path}')
  overlays.append(image)

# Set initial overlay
header = overlays[0]

# Drawing parameters
draw_color = (0, 0, 255)
brush_thickness = 20
eraser_thickness = 85

# Capture video and set resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Set up tracker
tracker = HandTracker.Tracker()
x_pos, y_pos = 0, 0
img_canvas = numpy.zeros((720, 1280, 3), numpy.uint8)

while True:
  # Capture image
  success, img = cap.read()
  img = cv2.flip(img, 1)

  # Identify hand landmarks
  img = tracker.track_hands(img)
  landmarks = tracker.get_landmarks(img, draw=False)

  if len(landmarks) > 0:
    # Get fingertip locations
    x1, y1 = landmarks[8][1:]
    x2, y2 = landmarks[12][1:]
    x3, y3 = landmarks[16][1:]
    x4, y4 = landmarks[20][1:]

    # Check which fingers are up
    fingers = tracker.read_fingers()

    # Select mode detected (Index and middle fingers are up)
    # When Selection Mode is engaged, reset x_pos and y_pos
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
      x_pos, y_pos = 0, 0  
      # Create input regions and corresponding controls
      if y1 < 125:
        if 200 < x1 < 400:
          header = overlays[0]
          draw_color = (0, 0, 255)
        elif 500 < x1 < 700:
          header = overlays[1]
          draw_color = (0, 255, 0)
        elif 750 < x1 < 900:
          header = overlays[2]
          draw_color = (255, 255, 0)
        elif 1000 < x1 <= 1200:
          header = overlays[3]
          draw_color = (0, 0, 0)
      cv2.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv2.FILLED)

    # Drawing mode detected (only the index finger is up)
    if fingers[1] == 1 and fingers[2] == 0:
      cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
      if x_pos == 0 and y_pos == 0:  # If we're in the very first frame
        x_pos, y_pos = x1, y1

      if draw_color == (0, 0, 0):
        cv2.line(img, (x_pos, y_pos), (x1, y1), draw_color, eraser_thickness)
        cv2.line(img_canvas, (x_pos, y_pos), (x1, y1),
                 draw_color, eraser_thickness)
      else:
        cv2.line(img, (x_pos, y_pos), (x1, y1), draw_color, brush_thickness)
        cv2.line(img_canvas, (x_pos, y_pos), (x1, y1),
                 draw_color, brush_thickness)

      x_pos, y_pos = x1, y1

  img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
  ret, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
  img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
  img = cv2.bitwise_and(img, img_inv)
  img = cv2.bitwise_or(img, img_canvas)

  # Setting initial header image
  img[0:125, 0:1280] = header
  cv2.imshow("FingerPainter", img)
  cv2.waitKey(1)
