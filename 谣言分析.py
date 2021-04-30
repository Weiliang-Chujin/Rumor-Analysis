from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score, \
recall_score,f1_score,cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pandas as pd
random.seed(2020)
df1 = pd.read_csv('DXYRumors.csv')
df2 = pd.read_csv('covid19_rumors.csv')

'''
操作一数据预处理
'''
#删除df1多余列
drop_list = ['_id','id','summary','body','sourceUrl']
df1.drop(drop_list,axis=1, inplace=True)
#连接两个数据
data = df1.append(df2).reset_index()
#将data的rumorType列转为连续的编号
df2[['rumorType']] = df2[['rumorType']].apply(LabelEncoder().fit_transform)
#查看数据前五行
print(data.head(5))
#时间转换，只取年月日
data['crawlTime'] = pd.to_datetime(data['crawlTime'])
data = data.sort_values(by='crawlTime')
for i,a in enumerate(data['crawlTime']):
    data.loc[i,'crawlTime'] = a.strftime("%Y/%m/%d")
#数据去重
print("去重之前数据形状：",data.shape)
data.drop_duplicates(keep='last',inplace=True)
print("去重之后数据形状：",data.shape)
#缺失值检测
print("data每个特征缺失值的数目为:\n",data.isnull().sum())

'''
操作二查看三种谣言数的具体分布
'''
#根据时间分组聚合查看每个时间点下的谣言数并重置索引
rumor_gp = data.groupby(['crawlTime'])['title'].agg('count').reset_index()
#将时间和每日谣言总数转成数组
x_time = []
y_count = []
for i,a in enumerate(rumor_gp['crawlTime']):
    x_time.append(a)
    y_count.append(rumor_gp.loc[i,'title'])
print("时间序列：\n",x_time)
print("每天总谣言数：\n",y_count)
#将每日三种谣言数转成数组
rumor_type = data.groupby(['crawlTime','rumorType'])['title'].agg('count').reset_index()
rumor_type = rumor_type.pivot(index='crawlTime',columns='rumorType',values='title')
rumor_type = rumor_type.rename_axis(None,axis=1).reset_index()
rumor_type = rumor_type.rename(columns = {0:'fake',1:'doubt',2:'true'})
rumor_type = rumor_type.fillna(0)
r_fake = []
r_doubt = []
r_true = []
for i,a in enumerate(rumor_type['fake']):
    r_fake.append(int(a))
    r_doubt.append(int(rumor_type.loc[i,'doubt']))
    r_true.append(int(rumor_type.loc[i, 'true']))
print("每天证实为假谣言数：\n",r_fake)
print("每天未被证实谣言数：\n",r_doubt)
print("每天证实为真谣言数：\n",r_true)
#计算三种谣言数量
fake = np.sum(np.array(r_fake))
doubt = np.sum(np.array(r_doubt))
true = np.sum(np.array(r_true))
print("三种谣言数（证实为假，未被证实，证实为真）：\n",fake,doubt,true)

'''
操作三MultinomialNB模型（多项式贝叶斯分类器）
'''
#用TfIdfVectorizer将文本向量化
zhTokenizer = jieba.cut
v = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",tokenizer=zhTokenizer,
lowercase=False,stop_words=['是','的'],max_features = 250)
y = data['rumorType']
#将谣言的内容和题目合一起
x_txt	= data[['mainSummary','title']].apply(lambda x:' '.join(x),axis=1)
#划分训练集和测试集
x_tr,x_te,y_tr,y_te = train_test_split(x_txt,y,test_size=0.2,stratify=y)
#构建模型并训练
x_tr_v = v.fit_transform(x_tr)
model_bl = mnb()
model_bl.fit(x_tr_v,y_tr.values)
x_te_v = v.transform(x_te)
y_pred = model_bl.predict(x_te_v)
#accuracy_score分类准确率，算出分类中正确分类的百分比
print("使用Multinomial Naive Bayes模型预测的准确率：",accuracy_score(y_te,y_pred))
print('使用Multinomial Naive Bayes模型预测的精确率为：',precision_score(y_te,y_pred, average="micro"))
print('使用Multinomial Naive Bayes模型预测的召回率为：',recall_score(y_te,y_pred, average="micro"))
print('使用Multinomial Naive Bayes模型预测的F1值为：',f1_score(y_te,y_pred, average="micro"))
print('使用Multinomial Naive Bayes模型预测的Cohen’s Kappa系数为：',cohen_kappa_score(y_te,y_pred))
print('使用Multinomial Naive Bayes模型预测的分类报告为：','\n',classification_report(y_te,y_pred))

#测试MultinomialNB的预测性能随alpha参数的影响
alphas = np.logspace(-2, 5, num=200)
train_scores = []
test_scores = []
for alpha in alphas:
    cls = mnb(alpha=alpha)
    cls.fit(x_tr_v, y_tr)
    train_scores.append(cls.score(x_tr_v, y_tr))
    test_scores.append(cls.score(x_te_v, y_te))
#绘图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(alphas, train_scores, label="Training Score")
ax.plot(alphas, test_scores, label="Testing Score")
ax.legend(['Training Score','Testing Score'])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("score")
ax.set_ylim(0, 1.0)
ax.set_title("MultinomialNB")
ax.set_xscale("log")
plt.show()

'''
操作四LinearSVC模型（线性分类支持向量机）
'''
content	= data[['mainSummary','title']].apply(lambda x:' '.join(x),axis=1)
train_x,test_x,train_y,test_y = train_test_split(content,data['rumorType'],test_size=0.2,random_state=42)
# TFIDF计算
model_tfidf = TFIDF(min_df=5, max_features=5000, ngram_range=(1,3), use_idf=1, smooth_idf=1)
model_tfidf.fit(train_x)
# 把文档转换成 X矩阵（该文档中该特征词出现的频次），行是文档个数，列是特征词的个数
train_vec = model_tfidf.transform(train_x)
# 模型训练
model_SVC = LinearSVC()
#概率校正
clf = CalibratedClassifierCV(model_SVC)
clf.fit(train_vec,train_y)
# 把文档转换成矩阵
test_vec = model_tfidf.transform(test_x)
pre_test = clf.predict(test_vec)
score = accuracy_score(test_y,pre_test)
print("使用LinearSVC模型预测的准确率:",score)
print('使用LinearSVC模型预测的精确率为：',precision_score(test_y,pre_test, average="micro"))
print('使用LinearSVC模型预测的召回率为：',recall_score(test_y,pre_test, average="micro"))
print('使用LinearSVC模型预测的F1值为：',f1_score(test_y,pre_test, average="micro"))
print('使用LinearSVC模型预测的Cohen’s Kappa系数为：',cohen_kappa_score(test_y,pre_test))
print('使用LinearSVC模型预测的分类报告为：','\n',classification_report(test_y,pre_test))


