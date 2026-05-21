# multi_view_zmq_client.py
import cv2
import zmq
import numpy as np

# ZMQ context and subscriber setup
context = zmq.Context()

# Each camera has its own ZMQ port from the server config
ports = {
    "cam_left": 55555,
    "cam_center": 55556,
    "cam_right": 55557,
}

# Create a socket for each port
sockets = {}
for name, port in ports.items():
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://127.0.0.1:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Receive all messages
    sockets[name] = socket

print("Connected to ZMQ streams. Press 'q' in any window to exit.")

while True:
    for name, socket in sockets.items():
        # Non-blocking receive with timeout to prevent freezing
        try:
            raw = socket.recv(flags=zmq.NOBLOCK)
            # Decode JPEG to OpenCV BGR image
            img_array = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow(name, frame)
        except zmq.Again:
            # No new frame available for this camera right now
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()