import os
import cv2

# 要被合成的多张图片所在文件夹
# 路径分隔符最好使用“/”,而不是“\”,“\”本身有转义的意思；或者“\\”也可以。
# 因为是文件夹，所以最后还要有一个“/”
file_dir = './det_results/ucf24/yowo/video_clips'
list = []
for root, dirs, files in os.walk(file_dir):
    for file in files:
        list.append(file)  # 获取目录下文件名列表

print(len(list))

# VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
# 'MJPG'意思是支持jpg格式图片
# fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
# (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
# 定义保存视频目录名称和压缩格式，像素为1280*720
video = cv2.VideoWriter('./test.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24,
                        (320, 240))

for i in range(1, len(list)):
    img = cv2.imread('./det_results/ucf24/yowo/video_clips/' + list[i - 1])  # 读取图片
    # img = cv2.resize(img,(1981,991)) #将图片转换为1280*720像素大小
    video.write(img)  # 写入视频

# 释放资源
video.release()

