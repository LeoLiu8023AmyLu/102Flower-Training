# -*- coding: utf-8 -*-

import skimage.io as io
import os
from sklearn.decomposition import PCA


picNum = 20 * (10 ** 4)  #根据实际内存数量可能需要改小


def readFile(ImageFile):
    mandrill = io.imread(ImageFile)
    picData = list(mandrill.flatten())
    currentLen = len(picData)
    if currentLen<picNum:
        add = [0 for i in range(picNum-currentLen)]
        picData = picData + add
    else:
        picData = picData[0:picNum]
    return picData


def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(train_x, train_y)
    return model


# 每个目录取5个图片训练PCA模型
picDic = []
pathDir = os.listdir('./data/train')
for subPath in pathDir:
    dirPath = './data/train/' + subPath
    subFile = os.listdir(dirPath)
    for filePath in subFile[0:3]:  #根据实际内存数量可能需要改小
        finalDir = dirPath+'/' + filePath
        picData = readFile(finalDir)
        picDic.append(picData)
print('len(picDic):',len(picDic))

# 使用PCA模型进行特征提取，构造 trainX trainY testX testY
pca=PCA(n_components=200)
pca.fit_transform(picDic)
del picDic
print('PCA model trained over.')

trainX=[]
trainY=[]
testX=[]
testY=[]
rawdata=[]
y=[]

print('trainX trainY ...')
pathDir = os.listdir('./data/train')
for subPath in pathDir:
    print('train subPath:', subPath)
    rawdata = []
    y = []
    dirPath = './data/train/' + subPath
    subFile = os.listdir(dirPath)
    for filePath in subFile:
        finalDir = dirPath+'/' + filePath
        y.append(int(subPath.split('.')[0]))
        rawdata.append(readFile(finalDir))
    pcaData = pca.transform(rawdata).tolist()
    trainX = trainX+pcaData
    trainY = trainY + y
print('trainX trainY over.')

print('testX testY ...')
pathDir = os.listdir('./data/validation')
for subPath in pathDir:
    print('test subPath:',subPath)
    rawdata = []
    y = []
    dirPath = './data/validation/' + subPath
    subFile = os.listdir(dirPath)
    for filePath in subFile:
        finalDir = dirPath+'/' + filePath
        y.append(int(subPath.split('.')[0]))
        rawdata.append(readFile(finalDir))
    pcaData = pca.transform(rawdata).tolist()
    testX = testX+pcaData
    testY = testY + y
print('testX testY over.')


# 计算 predictY
print('training GBDT model...') #随机森林算法
model = gradient_boosting_classifier(trainX, trainY)
print('using GBDT model...')
predictY = model.predict(testX).tolist()
print('predictY:',predictY)
print('testY:',testY)

# 计算 分类正确率acc
hit = 0
num = len(testY)
for i in range(num):
    if testY[i]==predictY[i]:
        hit+=1
acc = hit*1.0/num
print('acc:',acc)


