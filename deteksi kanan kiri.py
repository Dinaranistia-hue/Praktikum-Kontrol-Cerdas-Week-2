import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # 🔄 Mirror dulu
    img = cv2.flip(img, 1)

    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hasil = pose.process(imgrgb)

    if hasil.pose_landmarks:
        mp_draw.draw_landmarks(img, hasil.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = hasil.pose_landmarks.landmark
        h, w, c = img.shape

        # ===== TANGAN KIRI =====
        shoulder_left = landmarks[12]
        wrist_left = landmarks[16]

        cx_left = int(wrist_left.x * w)
        cy_left = int(wrist_left.y * h)

        if wrist_left.y < shoulder_left.y:
            cv2.putText(img, "Tangan Kiri Terangkat",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            cv2.circle(img, (cx_left, cy_left), 20, (0, 0, 255), -1)

        # ===== TANGAN KANAN =====
        shoulder_right = landmarks[11]
        wrist_right = landmarks[15]

        cx_right = int(wrist_right.x * w)
        cy_right = int(wrist_right.y * h)

        if wrist_right.y < shoulder_right.y:
            cv2.putText(img, "Tangan Kanan Terangkat",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            cv2.circle(img, (cx_right, cy_right), 20, (0, 0, 255), -1)

    cv2.imshow("Mirror Pose Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()