'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #统计每个类别出现的次数，存放在labelCounts里边。
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries   #概率
        shannonEnt -= prob * log(prob,2) #log base 2    #香农熵
    return shannonEnt
    
def splitDataSet(dataSet, axis, value): #返回划分后的数据集
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]        |#chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])|#这两行代码的作用是删掉axis，即按照划分数据的那个特征，这也说明了为什么一种退化情况是没特征了
            retDataSet.append(reducedFeatVec)#结果是retDataSet=[[...axis-1,axis+1,...],[...axis-1,axis+1,...],...,[...axis-1,axis+1,...]]
    return retDataSet                        #其中每个元素之前都满足featVec[axis]==value
    
def chooseBestFeatureToSplit(dataSet):  #谁造成的增益大就按照谁划分，按照谁划分决定了决策树每一层的判断节点是什么内容，每个特征谁在上，谁在下
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)   #原始数据的香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)      #get a set of unique values 转成集合就没有重复元素了，集合的内容是这个特征下的所有，可能取值，即树的分叉
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #这下splitDataSet的参数value就明确了，就是特征的取值，按特征划分就是特征不同的取值划分为不同的分类
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]   #？？？？

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]   | #每个实例的最后一列是类别
    if classList.count(classList[0]) == len(classList):| #如果类型列表的第一个元素个数等于列表长度，说明分类只有这一类了，即分类完成了。
        return classList[0]                            | #stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:                           | #一个实例的长度为1，说明只剩下类别了，没有特征了，即分类完成了。
        return majorityCnt(classList)                  | #两种退化情况
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels) #递归的创建树
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]  #树根
    secondDict = inputTree[firstStr]    #树除了树根
    featIndex = featLabels.index(firstStr)  #字符转化为索引
    key = testVec[featIndex]    #终于得到了测试数据的第一个特征的值
    valueOfFeat = secondDict[key]   #继续在决策树上下移了一层，得到第一个特征在测试数据的取值下得到的分类（叶节点）或下一个特征（判断节点）
    if isinstance(valueOfFeat, dict): #如果valueOfFeat的类型是字典，那还要继续深入
        classLabel = classify(valueOfFeat, featLabels, testVec) #否则返回分类
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
