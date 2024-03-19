
# Import necessary modules
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2
import os
import cvzone

# Initialize pose detector and webcam capture
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()
cap = cv2.VideoCapture(0)
shirtFolderPath = "resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 430 / 320
shirtRatioHeightWidth = 1495 / 1250

# Define the streaming view
@gzip.gzip_page
def video_feed(request):
    def frame_generator():
        while True:
            success, img = cap.read()
            img = detector.findPose(img)
            img = cv2.flip(img,1)
            lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
            if lmList:
                lm11 = lmList[11][1:3]
                lm12 = lmList[12][1:3]
                imgShirt = cv2.imread(os.path.join(shirtFolderPath,listShirts[0]),cv2.IMREAD_UNCHANGED)
                imgShirt = cv2.resize(imgShirt,(0,0),None,0.3,0.3)

                widthOfShirt = int(lm11[0]-lm12[0]) * fixedRatio
                # currentScale = (lm11[0] - lm12[0]) / 190
                # offset = int(180 * currentScale), int(210 * currentScale)
                try:
                    img = cvzone.overlayPNG(img, imgShirt, lm11)
                except Exception as error:
                    print(error)

            ret, jpeg = cv2.imencode('.jpg', img)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return StreamingHttpResponse(frame_generator(), content_type='multipart/x-mixed-replace; boundary=frame')

