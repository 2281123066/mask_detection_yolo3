from yolo import YOLO, detect_video
from PIL import Image
import os
import cv2


def detect_image_test(img_path, out_path):
    try:
        src_img = Image.open(img_path)
    except:
        print("open error, try again!")
    else:
        r_image = yolo.detect_image(src_img)
        yolo.close_session()

        r_image.save(os.path.join(out_path, '1.png'))


if __name__ == '__main__':
    yolo = YOLO()
    out_path = './output/'
    img_path = './test/1.jpg'
    video_path = './test/test4.mp4'
    detect_image_test(img_path, out_path) # 单张图片测试

    # detect_video(video_path, output_path=os.path.join(out_path, 'out_test4.mp4')) # 本地视频测试
    #  detect_video(video_path=None, output_path=""))  # 实时视频测试
