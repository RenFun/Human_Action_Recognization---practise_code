# Author:RenFun
# File:Video_pretreatment.py
# Time:2022/01/22

import os
import cv2
import csv

# 对视频文件进行预处理：将视频按帧分割成图片，然后再输入到网络中进行训练
# 定义常量
# 视频的路径
video_src_path = "D:\\DataSet\\KTH\\KTHVideo"
# 视频帧图像保存的路径
frame_des_path = "D:\\DataSet\\KTH\\KTHFrame"
# 建立一个CSV文件用于保存样本的路径：样本为二维数据，样本的保存路径和标签
train_save_path = "D:\\DataSet\\KTH\\Data&Label.csv"


# 视频转换为帧图像函数：将一个视频按帧进行转换，一帧为一个图像
# 参数：视频路径，视频帧图像保存路径，时间间隔，图片数量，文件名（标签）
def VideoToFrame(videopath, framepath, interval, count, filelabel):
    # 由视频地址获取本地的视频
    capture = cv2.VideoCapture(videopath)  # .VideoCapture(视频路径)
    # 输出视频的信息
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    num_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # count = 0
    # 160.0 120.0 25.0 360.0，分别是视频的高，宽，帧数，视频总帧数（帧数*视频时长，也就是一个视频能够转换为帧图像的数量）
    print(width, height, fps, num_frames)
    # 按帧读取视频：返回值为bool型（正确读取则返回true，反之false）；frame为每一帧的图像，是三维矩阵，即（height，width，channels），使用BGR格式，即blue，green，red
    success, frame = capture.read()
    # 用count记录一个视频能转化为多少帧图像
    count += 1
    # 只要success为真，就一直进行操作，即按帧读取视频直至结束
    while success:
        if count % interval == 0:
            # print("Writing the number %d of frame to src file" % (count//interval))
            # 将读取的图像转换为灰度图，即将BGR格式转换为GRAY格式
            GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 将一帧图片保存的地址和标签写入csv的一行，地址为视频地址+视频名+帧图像数.jpg；标签就是视频名
            train_file.writerow([framepath + '%d.jpg' % count, filelabel])
            # 保存一帧图片：地址为视频地址+视频名+帧图像数.jpg，以灰度图格式存储
            cv2.imwrite(framepath + '%d.jpg' % (count // interval), GRAY)  # 取整除 count//interval
        success, frame = capture.read()
        count += 1
    capture.release()
    # print("帧图像数量：", count)
    print("Encoding file %s success!" % file)
    # 返回帧图像数量
    return count


# 获取视频数据集文件夹下的文件（名）
files = os.listdir(video_src_path)                      # 返回指定路径下所包含的文件
# 打开.csv文件并写入，如果该文件不存在则先创建再打开
file_train = open(train_save_path, 'a', newline='')     # newline=''表示避免出现空行
train_file = csv.writer(file_train, dialect='excel')

# csv_file.writerow(['Mes','Label'])


# 遍历files：遍历视频数据集文件夹里的每个文件（名）
for file in files:
    # 组合形成文件路径：原视频数的路径+视频数据集文件夹中的文件名，即每个视频文件的路径
    file_to_video = os.path.join(video_src_path, file)
    # 获取每个视频的具体路径的列表并分别遍历
    # videos = os.listdir(file_to_video)
    videos = files
    # 如果没有则创建文件夹，文件夹名为原视频数据集文件路径+视频名
    if not os.path.isdir(frame_des_path + file):
        os.mkdir(frame_des_path + file)
    # 帧图像保存路径，进入该目录文件夹下用于保存文件
    frame_save_path = frame_des_path + file + '\\'            # 两个反斜杠是为了区分转义符
    index = 0
    # for video in videos:
    #     # video_cur_path = os.path.join(file_to_video, video)  # 每一个视频的地址
    #     video_cur_path = file_to_video
    #     count = VideoToFrame(video_cur_path, frame_save_path, 1, index, file)
    #     index = count
    video_cur_path = file_to_video
    count = VideoToFrame(video_cur_path, frame_save_path, 1, index, file)
    index = count
# 操作结束，关闭文件
file_train.close()
