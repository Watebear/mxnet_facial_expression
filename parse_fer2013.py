# copy from： https://www.jianshu.com/p/edbffccb3743
import pandas as pd
import numpy as np
import os
from PIL import Image

# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

emotions = {
    '0':'angry', #生气
    '1':'disgust', #厌恶
    '2':'fear', #恐惧
    '3':'happy', #开心
    '4':'sad', #伤心
    '5':'surprised', #惊讶
    '6':'neutral', #中性
}

#创建文件夹
def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)

def saveImageFromFer2013(file, base_dir='datasets/Fer2013'):
    #读取csv文件
    faces_data = pd.read_csv(file)
    imageCount = 0
    #遍历csv文件内容，并将图片数据按分类保存
    for index in range(len(faces_data)):
        #解析每一行csv文件内容
        emotion_data = faces_data.loc[index][0]
        image_data = faces_data.loc[index][1]
        usage_data = faces_data.loc[index][2]
        #将图片数据转换成48*48
        data_array = list(map(float, image_data.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)

        #选择分类，并创建文件名
        dirName = usage_data
        emotionName = emotions[str(emotion_data)]

        #图片要保存的文件夹
        imagePath = os.path.join(base_dir, dirName, emotionName)

        # 创建“表情”文件夹
        createDir(imagePath)

        #图片文件名
        imageName = os.path.join(imagePath, str(index) + '.jpg')

        image_file = Image.fromarray(image).convert('L')
        image_file.save(imageName)
        imageCount = index
    print('总共有' + str(imageCount) + '张图片')


if __name__ == '__main__':
    saveImageFromFer2013('datasets/Fer2013/fer2013.csv')