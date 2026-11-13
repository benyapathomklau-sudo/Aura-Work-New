import time
import cv2
import mediapipe as mp
import numpy as np
import winsound

# ---------------------------------------------------------
# 1. ฟังก์ชันคำนวณมุมองศา (หู, ไหล่, สะโพก)
# ---------------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)  # หู
    b = np.array(b)  # ไหล่ (จุดหมุน)
    c = np.array(c)  # สะโพก

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# ---------------------------------------------------------
# 2. ตัวแปรสำหรับตั้งค่าและเก็บสถิติ (Pomodoro & Analytics)
# ---------------------------------------------------------
FOCUS_DURATION = 25 * 60  # ระยะเวลาทำงาน (25 นาที)
BREAK_DURATION = 5 * 60   # ระยะเวลาพัก (5 นาที)

start_time = time.time()
mode = "FOCUS"  # สถานะ: "FOCUS" หรือ "BREAK"

# สถิติสำหรับสรุปผลประจำวัน (Daily Report)
total_frames = 0
good_posture_frames = 0
bad_posture_frames = 0
break_count = 0

# ---------------------------------------------------------
# 3. เริ่มต้นเปิดใช้งาน MediaPipe & Camera
# ---------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

print("=" * 50)
print("  AuraWork Guardian - Holistic Well-being System")
print("  กด 'q' ที่หน้าต่างกล้องเพื่อปิดระบบและดูสรุปผลประจำวัน")
print("=" * 50)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("ไม่สามารถเปิดใช้งานกล้องได้")
        break

    # แปลงสีภาพจาก BGR เป็น RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    current_time = time.time()
    elapsed_time = current_time - start_time

    # ---------------------------------------------------------
    # 4. ระบบสลับโหมด Pomodoro+เสียงทุ้มพิเศษ (Focus / Break)
    # ---------------------------------------------------------
    if mode == "FOCUS" and elapsed_time >= FOCUS_DURATION:
        mode = "BREAK"
        start_time = time.time()
        break_count += 1
        winsound.Beep(350, 250)  # ปรับเป็น ความถี่ 500 Hz (เสียงทุ้มต่ำนุ่มนวลมาก), ดัง 150 มิลลิวินาที
    elif mode == "BREAK" and elapsed_time >= BREAK_DURATION:
        mode = "FOCUS"
        start_time = time.time()
        winsound.Beep(420, 200)  # ปรับเป็น ความถี่ 400 Hz (เสียงทุ้ม นุ่มนวลขึ้นอีกนิด)

    # ---------------------------------------------------------
    # 5. การตรวจจับและประมวลผลร่างกาย (Physical & Mental Health)
    # ---------------------------------------------------------
    status = "Good Posture"
    color = (0, 255, 0)
    warning_msg = ""

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        image_height, image_width, _ = image.shape

        try:
            # พิกัด หู ไหล่ สะโพก (ฝั่งขวา)
            ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * image_width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * image_height]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height]

            # 5.1 ตรวจจับระยะห่างหน้าจอ (Screen Distance Check)
            # ใช้ขนาดความกว้างระหว่างไหล่ 2 ข้างประมาณความใกล้-ไกล
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_width = np.abs(left_shoulder.x - right_shoulder.x) * image_width
            
            if shoulder_width > image_width * 0.45: # ถ้านั่งใกล้จอเกินไป
                warning_msg = "TOO CLOSE TO SCREEN!"

            # 5.2 คำนวณมุมหลังค่อม
            angle = calculate_angle(ear, shoulder, hip)
            total_frames += 1

            if angle < 140:
                status = "Bad Posture (Slouching!)"
                color = (0, 0, 255)
                bad_posture_frames += 1
                winsound.Beep(300, 80)  # ปรับเป็น ความถี่ 600 Hz (ทุ้มต่ำมากๆ), สั้นแค่ 100 มิลลิวินาที
            else:
                good_posture_frames += 1

            # แสดงตัวเลของศาบนหน้าจอ
            cv2.putText(image, f'Angle: {int(angle)}',
                        (int(shoulder[0]) - 50, int(shoulder[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # วาดเส้นโครงร่างร่างกาย
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            pass

    # ---------------------------------------------------------
    # 6. แสดงผล UI บนหน้าต่างกล้อง
    # ---------------------------------------------------------
    # แสดงคำเตือนระยะห่างหน้าจอ
    if warning_msg:
        cv2.putText(image, warning_msg, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    # โหมด FOCUS
    if mode == "FOCUS":
        remaining = int(FOCUS_DURATION - elapsed_time)
        mins, secs = divmod(remaining, 60)
        cv2.putText(image, f'FOCUS TIME: {mins:02d}:{secs:02d}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, status, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # โหมด BREAK (คำแนะนำพักสายตาและยืดเส้นยืดสาย)
    else:
        remaining = int(BREAK_DURATION - elapsed_time)
        mins, secs = divmod(remaining, 60)
        # ฉากหลังสีทึบสำหรับโหมดพัก
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image_width, image_height), (50, 50, 50), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        cv2.putText(image, f'--- TIME TO BREAK & RECHARGE ({mins:02d}:{secs:02d}) ---', 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "1. Stretch Neck & Shoulders (ยืดคอ-ไหล่)", 
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "2. Look 20 feet away for 20s (มองไกลพักสายตา)", 
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "3. Take 3 Deep Breaths (ฝึกหายใจเข้าลึกๆ)", 
                    (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # แสดงหน้าต่างโปรแกรม
    cv2.imshow('AuraWork Guardian - Holistic Well-being', image)

    # กดปุ่ม 'q' เพื่อปิดโปรแกรม
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# 7. สรุปผลสถิติประจำวัน (Daily Ergonomic & Health Summary)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("        📊 DAILY HEALTH & PRODUCTIVITY REPORT")
print("=" * 50)
if total_frames > 0:
    good_pct = (good_posture_frames / total_frames) * 100
    bad_pct = (bad_posture_frames / total_frames) * 100
    print(f"• สัดส่วนท่านั่งที่ดี (Good Posture): {good_pct:.1f}%")
    print(f"• สัดส่วนหลังค่อม (Slouching):      {bad_pct:.1f}%")
else:
    print("• ไม่พบข้อมูลการตรวจจับท่าทาง")

print(f"• จำนวนครั้งที่พักยืดเส้นยืดสาย (Breaks):  {break_count} ครั้ง")
if total_frames > 0 and good_pct >= 80:
    print("🌟 ข้อแนะนำ: วันนี้คุณดูแลสุขภาพท่านั่งได้เยี่ยมมาก!")
else:
    print("💡 ข้อแนะนำ: ลองปรับระดับความสูงของจอคอมพิวเตอร์เพื่อลดอาการหลังค่อมนะครับ")
print("=" * 50 + "\n")
