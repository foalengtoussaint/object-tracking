import cv2
cap = cv2.VideoCapture(4)  # or 0, 4
if not cap.isOpened():
    print("Failed to open camera 8")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_frame.jpg", frame)
        print("Success! Frame saved.")
    cap.release()