

import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv
import os


predictor_path = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

path_read = "/Users/zhaoyuyin/Documents/CISC499/Project_Code/dataset/test/yawning"

for file_name in os.listdir(path_read):

    aa=(path_read +"/"+file_name)

    img=cv2.imdecode(np.fromfile(aa, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if img is not None:

        img_shape=img.shape
        img_height=img_shape[0]
        img_width=img_shape[1]

        path_save="/Users/zhaoyuyin/Documents/CISC499/Project_Code/dataset/test/yawning1"

        dets = detector(img,1)

        for k, d in enumerate(dets):

            if len(dets)>1:
                continue

            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])


            height = d.bottom()-d.top()
            width = d.right()-d.left()


            img_blank = np.zeros((height, width, 3), np.uint8)

            for i in range(height):
                if d.top()+i>=img_height:# 防止越界
                    continue
                for j in range(width):
                    if d.left()+j>=img_width:# 防止越界
                        continue
                    img_blank[i][j] = img[d.top()+i][d.left()+j]



            img_blank = cv2.resize(img_blank, (200, 200), interpolation=cv2.INTER_CUBIC)

            cv2.imencode('.png', img_blank)[1].tofile(path_save+"/"+file_name)

    else:
        print(file_name)
