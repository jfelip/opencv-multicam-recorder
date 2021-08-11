import cv2


class V4L2CameraDetector:
    def __init__(self, ncameras=50):
        for id in range(ncameras):
            cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"Valid capture device found with id: {id}")
                res, img = cap.read()
                if res:
                    print(f" - Capturing images: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
                          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}@{cap.get(cv2.CAP_PROP_FPS)}")
                else:
                    print(f" - NOT Capturing images: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
                          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}@{cap.get(cv2.CAP_PROP_FPS)}")
            cap.release()


if __name__ == "__main__":
    V4L2CameraDetector()
