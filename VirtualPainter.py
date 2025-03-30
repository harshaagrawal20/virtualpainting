import cv2
import numpy as np
import os
import HandTracking as htm

# Brush and Eraser Thickness
brushThickness = 25
eraserThickness = 100

# Define Colors
colorList = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]  # Pink, Blue, Green
drawColor = colorList[0]  # Default: Pink
selectedIndex = 0  # Default selection index

# Shape Selection
shapeList = ['None', 'Circle', 'Square', 'Star', 'Heart']
selectedShape = 'None'

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = htm.handDetector(detectionCon=0.7, maxHands=1)
xp, yp = 0, 0  # Previous points for drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Undo/Redo stacks
undo_stack = []
redo_stack = []

def save_to_undo():
    # Save the current canvas to the undo stack
    undo_stack.append(imgCanvas.copy())

def undo():
    # Pop from the undo stack and push to redo stack
    if undo_stack:
        redo_stack.append(imgCanvas.copy())  # Save the current state before undo
        imgCanvas[:] = undo_stack.pop()

def redo():
    # Pop from the redo stack and restore to canvas
    if redo_stack:
        save_to_undo()  # Save the current state to the undo stack before redoing
        imgCanvas[:] = redo_stack.pop()

while True:
    # Capture Image
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # Flip image for natural movement
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # Draw color selection circles with highlighting
    positions = [(150, 50), (300, 50), (450, 50), (600, 50)]  # Pink, Blue, Green, Eraser
    for i, color in enumerate(colorList):
        thickness = -1 if i == selectedIndex else 5  # Highlight selected color
        cv2.circle(img, positions[i], 30, color, thickness)
    cv2.rectangle(img, (570, 20), (630, 80), (0, 0, 0), -1 if selectedIndex == 3 else 5)  # Eraser

    # Draw shape selection with actual shapes
    shape_positions = [(750, 50), (850, 50), (950, 50), (1050, 50), (1150, 50)]
    for i, (cx, cy) in enumerate(shape_positions):
        thickness = -1 if shapeList[i] == selectedShape else 2
        # Draw shapes instead of text
        if shapeList[i] == 'Circle':
            cv2.circle(img, (cx, cy), 30, (255, 255, 255), 3)  # Draw Circle
        elif shapeList[i] == 'Square':
            cv2.rectangle(img, (cx - 30, cy - 30), (cx + 30, cy + 30), (255, 255, 255), 3)  # Draw Square
        elif shapeList[i] == 'Star':
            pts = np.array([[cx, cy - 30], [cx - 15, cy - 10], [cx - 30, cy - 10], [cx - 20, cy + 10],
                            [cx - 25, cy + 30], [cx, cy + 15], [cx + 25, cy + 30], [cx + 20, cy + 10],
                            [cx + 30, cy - 10], [cx + 15, cy - 10]], np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)  # Draw Star
        elif shapeList[i] == 'Heart':
            heart_pts = np.array([[cx, cy - 20], [cx - 15, cy - 40], [cx - 30, cy - 20], [cx - 15, cy],
                                  [cx, cy + 15], [cx + 15, cy], [cx + 30, cy - 20], [cx + 15, cy - 40]], np.int32)
            cv2.polylines(img, [heart_pts], isClosed=True, color=(255, 255, 255), thickness=3)  # Draw Heart
        cv2.rectangle(img, (cx - 30, cy - 30), (cx + 30, cy + 30), (255, 255, 255), thickness)

    if lmList and len(lmList) > 12:
        try:
            x1, y1 = lmList[8][1:]  # Index Finger Tip
            x2, y2 = lmList[12][1:]  # Middle Finger Tip

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Selection Mode: Two fingers up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0  # Reset previous points

                # Check for color selection
                for i, (cx, cy) in enumerate(positions):
                    if (x1 - cx) ** 2 + (y1 - cy) ** 2 < 900:  # Inside Circle
                        drawColor = colorList[i]
                        selectedIndex = i

                # Check for eraser selection
                if 570 < x1 < 630 and 20 < y1 < 80:
                    drawColor = (0, 0, 0)  # Eraser
                    selectedIndex = 3

                # Check for shape selection
                for i, (cx, cy) in enumerate(shape_positions):
                    if cx - 30 < x1 < cx + 30 and cy - 30 < y1 < cy + 30:
                        selectedShape = shapeList[i]

            # Drawing Mode: Index Finger Only
            elif fingers[1] and not fingers[2]:
                if selectedShape == 'None':
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                    xp, yp = x1, y1
                else:
                    # Drawing selected shape
                    if selectedShape == 'Circle':
                        cv2.circle(imgCanvas, (x1, y1), 50, drawColor, 5)

                    elif selectedShape == 'Square':
                        cv2.rectangle(imgCanvas, (x1 - 50, y1 - 50), (x1 + 50, y1 + 50), drawColor, 5)

                    elif selectedShape == 'Star':
                        star_points = np.array([
                            [x1, y1 - 50], [x1 - 20, y1 - 20], [x1 - 50, y1 - 20], [x1 - 30, y1 + 5],
                            [x1 - 40, y1 + 40], [x1, y1 + 20], [x1 + 40, y1 + 40], [x1 + 30, y1 + 5],
                            [x1 + 50, y1 - 20], [x1 + 20, y1 - 20]
                        ], np.int32)
                        cv2.fillPoly(imgCanvas, [star_points], drawColor)  # Filled instead of outline




                    elif selectedShape == 'Heart':

                        heart_contour = np.array([

                            (x1, y1 + 20), (x1 - 35, y1 - 20), (x1 - 20, y1 - 60),

                            (x1, y1 - 40), (x1 + 20, y1 - 60), (x1 + 35, y1 - 20)

                        ], np.int32)

                        heart_contour = heart_contour.reshape((-1, 1, 2))  # Reshape for OpenCV compatibility

                        cv2.fillPoly(imgCanvas, [heart_contour], drawColor)  # Draw filled heart shape

            else:
                xp, yp = 0, 0

            # Clear Canvas when all fingers are up
            if all(f == 1 for f in fingers):
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            # Undo functionality: Three fingers down (Index, Middle, Ring)
            if not fingers[1] and not fingers[2] and not fingers[3]:  # Three fingers down
                undo()

            # Redo functionality: Palm Gesture (All fingers up)
            if all(f == 1 for f in fingers[1:]):  # Palm gesture (All fingers up)
                redo()

        except IndexError:
            pass  # Avoids crashes if lmList is unexpectedly short

    # Show the image and canvas
    img[0:720, 0:1280] = cv2.addWeighted(img[0:720, 0:1280], 0.5, imgCanvas[0:720, 0:1280], 0.5, 0)

    cv2.imshow("Shape Drawing", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
