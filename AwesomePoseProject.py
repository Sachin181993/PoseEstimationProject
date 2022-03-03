import cv2
import time
import PoseModule as pm


cap = cv2.VideoCapture('FootballPoseVideos/1.mp4')
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[15])
        cv2.circle(img, (lmList[15][1], lmList[15][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(1)
