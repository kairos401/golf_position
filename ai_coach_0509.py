import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_path = '01.mp4'
#'01.mp4'
#r'C:\Users\user\Desktop\Project\Project_3rd\bbanghyung\golf-ai-coach\videos_160\164.mp4'
#r'C:\Users\user\Desktop\Project\Project_3rd\bbanghyung\golf-ai-coach\videos_160\231.mp4'
#r'C:\Users\user\Desktop\Project\Project_3rd\bbanghyung\golf-ai-coach\videos_160\272.mp4'
#r'C:\Users\user\Desktop\Project\Project_3rd\bbanghyung\golf-ai-coach\videos_160\493.mp4'
cap = cv2.VideoCapture(video_path)

# 원본 영상의 너비와 높이를 얻습니다.
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 너비를 480으로 고정하고, 비율에 맞는 높이를 계산합니다.
new_width = 480
new_height = int(original_height * (new_width / original_width))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))
# confidence : 기본값 0.5이상이면 감지하게설정한것 같음
# model_complexity : 기본값 1 옵션(0,1,2), 숫자 높을수록 복잡한 모델 사용. 정확도 올라감
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2)

is_first = True # 어드레스 시 첫 프레임을 받아오기 위한 플래그
# 어드레스 시 첫 프레임의 좌표를 저장할 변수
paused = False
first_center_x, first_center_y, first_radius = None, None, None

while cap.isOpened():
    if not paused:
        ret, img = cap.read()
        if not ret:
            break
    img = cv2.resize(img, (new_width, new_height))
    img_h, img_w, _ = img.shape
    img_result = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)
    mp_drawing.draw_landmarks(
        img_result,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if results.pose_landmarks:
        # https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
        landmark = results.pose_landmarks.landmark
        left_ear_x = landmark[mp_pose.PoseLandmark.LEFT_EAR].x * img_w  #landmark 0~1 사잇값
        left_ear_y = landmark[mp_pose.PoseLandmark.LEFT_EAR].y * img_h
        right_ear_x = landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * img_w
        right_ear_y = landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * img_h
        center_x = int((left_ear_x + right_ear_x) / 2)
        center_y = int((left_ear_y + right_ear_y) / 2)
        radius = int((left_ear_x - right_ear_x) / 2)
        radius = max(radius, 20)

        right_wrist_x = landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * img_w
        right_wrist_y = landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * img_h
        right_elbow_x = landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * img_w
        right_elbow_y = landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * img_h
        right_shoulder_x = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img_w
        right_shoulder_y = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img_h
        # 좌표 추출
        right_wrist = np.array([right_wrist_x, right_wrist_y])
        right_elbow = np.array([right_elbow_x, right_elbow_y])
        right_shoulder = np.array([right_shoulder_x, right_shoulder_y])
        # 벡터 계산
        vector_shoulder_to_elbow = right_shoulder - right_elbow
        vector_elbow_to_wrist = right_wrist - right_elbow
        # 벡터의 크기(길이) 계산
        length_shoulder_to_elbow = np.linalg.norm(vector_shoulder_to_elbow)
        length_elbow_to_wrist = np.linalg.norm(vector_elbow_to_wrist)
        # 벡터의 내적 및 안전한 코사인 값 계산
        dot_product = np.dot(vector_shoulder_to_elbow, vector_elbow_to_wrist)
        cos_theta = dot_product / (length_shoulder_to_elbow * length_elbow_to_wrist)
        cos_theta = np.clip(cos_theta, -1, 1)  # 클리핑을 통해 값의 범위를 -1과 1 사이로 제한
        # 각도 계산 (라디안에서 도로 변환)
        angle = np.arccos(cos_theta)
        right_arm_angle = abs(np.degrees(angle))
        # print('Right arm angle:', right_arm_angle)
        
        left_eye_x = landmark[mp_pose.PoseLandmark.LEFT_EYE].x * img_w
        left_eye_y = landmark[mp_pose.PoseLandmark.LEFT_EYE].y * img_h
        right_eye_x = landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * img_w
        right_eye_y = landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * img_h
        nose_x = landmark[mp_pose.PoseLandmark.NOSE].x * img_w
        nose_y = landmark[mp_pose.PoseLandmark.NOSE].y * img_h
        head_top_x = 2*left_eye_x + 2*right_eye_x - 3*nose_x
        head_top_y = 2*left_eye_y + 2*right_eye_y - 3*nose_y
        right_hip_x = landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * img_w
        right_hip_y = landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img_h
        right_foot_index_x = landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * img_w
        right_foot_index_y = landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * img_h
        # 벡터 생성
        vector_head_top_to_right_hip = np.array([right_hip_x - head_top_x, right_hip_y - head_top_y])
        vector_right_foot_index_to_right_hip = np.array([right_hip_x - right_foot_index_x, right_hip_y - right_foot_index_y])
        # 벡터의 길이 계산
        length_head_top_to_right_hip = np.linalg.norm(vector_head_top_to_right_hip)
        length_right_foot_index_to_right_hip = np.linalg.norm(vector_right_foot_index_to_right_hip)
        # 벡터의 내적 계산
        dot_product_waist = np.dot(vector_head_top_to_right_hip, vector_right_foot_index_to_right_hip)
        # 코사인 각도 계산
        cos_theta_waist = dot_product_waist / (length_head_top_to_right_hip * length_right_foot_index_to_right_hip)
        cos_theta_waist = np.clip(cos_theta_waist, -1, 1)  # 값의 범위를 -1과 1 사이로 제한
        # 각도 계산 (라디안에서 도로 변환)
        angle_waist = np.arccos(cos_theta_waist)
        right_waist_angle = abs(np.degrees(angle_waist))
        
        # right wrist와 right shoulder 사이의 벡터 생성
        vector_shoulder_to_wrist = np.array([right_wrist_x - right_shoulder_x, right_wrist_y - right_shoulder_y])
        # 지면과의 각도 계산 (수평선과의 각도)
        horizontal_vector = np.array([1, 0])  # 완전한 수평 벡터
        dot_product = np.dot(vector_shoulder_to_wrist, horizontal_vector)
        length_shoulder_to_wrist = np.linalg.norm(vector_shoulder_to_wrist)
        cos_theta = dot_product / length_shoulder_to_wrist
        angle_with_horizontal = int(np.degrees(np.arccos(cos_theta)))  # 수평선과의 각도
        
        # 이미지에 각도 표시
        if is_first: # 어드레스 시 첫 프레임의 머리 좌표 저장
            first_center_x = center_x
            first_center_y = center_y
            first_radius = int(radius * 2)
            is_first = False
        else:
            cv2.circle(img_result, center=(first_center_x, first_center_y),
                radius=first_radius, color=(0, 255, 255), thickness=2)
            color = (0, 255, 0) # 초록색

            # 머리가 원래 위치보다 많이 벗어난 경우
            if center_x - radius < first_center_x - first_radius \
                or center_x + radius > first_center_x + first_radius:
                color = (0, 0, 255) # 빨간색
            # 오른팔 각도가 170도 이상인경우 good자세
            cv2.circle(img_result, (int(head_top_x), int(head_top_y)), 5, (0, 255, 255), -1)
            cv2.circle(img_result, center=(center_x, center_y),radius=radius, color=color, thickness=2)
            if right_arm_angle>170:
                color=(0,255,0) # 초록색
            else:
                color=(0,0,255)
            cv2.putText(img_result, f'Right Arm Angle : {right_arm_angle:.2f}', (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, color=color) #각도 구해서 format에 넣기
            # 오른 허리 각도가 155도 이하인 경우 good자세
            color_waist = (0, 255, 0) if right_waist_angle < 155 else (0, 0, 255)
            cv2.putText(img_result, f'Right Waist Angle : {right_waist_angle:.2f}', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, color_waist)
            cv2.putText(img_result, f'Best Angle : {angle_with_horizontal:.2f}', (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.75, color_waist)
            
            # 각도가 30도 근처일 때 이미지 캡처
            if (25 <= angle_with_horizontal <= 35) and (right_wrist_y > right_shoulder_y):
                cv2.imwrite(f'{video_path.split('.')[0]}_captured_frame.jpg', img_result)  # 이미지 저장

    cv2.imshow('AI Golf Coach', img_result)
    out.write(img_result)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):
        paused = True
    elif key == ord('b'):
        paused = False

pose.close()
cap.release()
out.release()
