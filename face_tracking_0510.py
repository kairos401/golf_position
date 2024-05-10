import cv2
import dlib
import time
from pymycobot.mycobot import MyCobot
from imutils import face_utils
import numpy as np

# Initialize the MyCobot instance
mc = MyCobot('COM8', 115200)
mc.set_gripper_mode(0)
mc.init_eletric_gripper()

# Function to ensure the robot has finished moving before continuing
def arm_mov_chk():
    time.sleep(0.6)
    while mc.is_moving():
        time.sleep(0.1)

# Function to move the robot arm based on an angle value
def move_robot_arm3(angle):
    """Move the robot arm according to the given angle"""
    mc.send_angle(4, angle, 20)
    # arm_mov_chk()

# Initialize the arm position
y_angle = 5
mc.send_angles([0, 0, 0, 0, -90, 0], 20)
arm_mov_chk()
mc.send_angles([0, 113, -125, y_angle, -90, 0], 20)
arm_mov_chk()

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Initialize video capture (0 for the primary webcam)
cap = cv2.VideoCapture(1)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cx = original_width // 2
frame_cy = original_height // 2


# Set the time interval in seconds (0.1 second interval)
interval = 0.1
last_move_time = time.time()  # Initialize with the current time

# Control loop to detect faces and adjust robot arm
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Determine the center of the detected faces and adjust the arm accordingly
    if faces:
        (x, y, w, h) = face_utils.rect_to_bb(faces[0])
        face_cx = x + w // 2
        face_cy = y + h // 2

        # Calculate vertical displacement
        gap_y = frame_cy - face_cy

        # If the displacement is significant, adjust the angle
        current_time = time.time()
        if current_time - last_move_time > interval:
            # If the displacement is significant, adjust the angle
            if abs(gap_y) > 50:
                if gap_y > 0:
                    y_angle -= 1  # Adjust upward
                    print("각도 증가")
                else:
                    y_angle += 1  # Adjust downward
                    print("각도 감소")

                # Move the robot arm and update the last move time
                y_angle = np.clip(y_angle, -20, 25)
                print(y_angle)
                move_robot_arm3(y_angle)
                last_move_time = current_time

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Dlib Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
