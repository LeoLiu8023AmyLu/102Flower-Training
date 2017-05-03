# 102Flower-Training </br>
机器学习练习，使用SVM、随机森林、Adaboost算法训练 102 flower 训练集 得到特征提取 </br>

## 相关资料 </br>
* [随机森林Matlab算法：http://blog.csdn.net/jiandanjinxin/article/details/51003840](http://blog.csdn.net/jiandanjinxin/article/details/51003840)</br>
* [随机森林 Python sklearn:http://blog.csdn.net/a819825294/article/details/51177435](http://blog.csdn.net/a819825294/article/details/51177435)</br>
* [matlab实现hog+svm图像二分类  http://blog.csdn.net/jcy1009015337/article/details/53763484](http://blog.csdn.net/jcy1009015337/article/details/53763484)</br>
* [随机森林  http://blog.csdn.net/dan1900/article/details/39030867](http://blog.csdn.net/dan1900/article/details/39030867)</br>
* [Matlab使用技巧  http://blog.csdn.net/lqhbupt/article/details/20292113](http://blog.csdn.net/lqhbupt/article/details/20292113)</br>
* [随机森林 介绍  http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#intro](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#intro)</br>
* [随机森林 原理 http://www.cnblogs.com/maybe2030/p/4585705.html](http://www.cnblogs.com/maybe2030/p/4585705.html)</br>
* [随机森林（原理/样例实现/参数调优） http://blog.csdn.net/y0367/article/details/51501780](http://blog.csdn.net/y0367/article/details/51501780)</br>

## 工作记录 </br>
* 2017.04.24 谈合作细节，了解项目相关领域知识(SVM CNN RF Adaboost等) 商讨价格，第一次优化 SVM 程序 </br>
* 2017.04.25 尝试完成 RF 第一版本程序</br>
* 2017.04.26 第二次优化 SVM 程序</br>
* 2017.04.27 完善 RF 程序 得到 最佳参数： T 100 M 128 </br>
* 2017.04.28 交接工作 讲解程序思想 </br>
* 2017.05.03 先打1500块钱  作为 第一部分交付额 </br>

## Matlab 分类器</br>

* MATLAB中的分类器 </br>

目前了解到的MATLAB中分类器有：K近邻分类器，随机森林分类器，朴素贝叶斯，集成学习方法，鉴别分析分类器，支持向量机。现将其主要函数使用方法总结如下，更多细节需参考MATLAB 帮助文件。</br>

设</br>

	训练样本：train_data             % 矩阵，每行一个样本，每列一个特征
	训练样本标签：train_label       % 列向量
	测试样本：test_data
	测试样本标签：test_label

* K近邻分类器 （KNN） </br>
	
	
	mdl = ClassificationKNN.fit(train_data,train_label,'NumNeighbors',1);
	predict_label = predict(mdl, test_data);
	accuracy = length(find(predict_label == test_label))/length(test_label)*100
               
 
* 随机森林分类器（Random Forest）</br>
	
	
	B = TreeBagger(nTree,train_data,train_label);
	predict_label = predict(B,test_data);
 
 
* 朴素贝叶斯 （Na?ve Bayes） </br>


	nb = NaiveBayes.fit(train_data, train_label);
	predict_label = predict(nb, test_data);
	accuracy = length(find(predict_label == test_label))/length(test_label)*100;
 
* 集成学习方法（Ensembles for Boosting, Bagging, or Random Subspace） <br>

	
	ens = fitensemble(train_data,train_label,'AdaBoostM1' ,100,'tree','type','classification');
	predict_label   =       predict(ens, test_data);
 
 
* 鉴别分析分类器（discriminant analysis classifier）</br>


	obj = ClassificationDiscriminant.fit(train_data, train_label);
	predict_label   =       predict(obj, test_data);
 
 
* 支持向量机（Support Vector Machine, SVM） </br>


	SVMStruct = svmtrain(train_data, train_label);
	predict_label  = svmclassify(SVMStruct, test_data)