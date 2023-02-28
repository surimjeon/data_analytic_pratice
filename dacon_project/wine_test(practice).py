# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
import pandas as pd

train=pd.read_csv('C:/Users/전수림/Desktop/데이터분석/와인품질분류/data/train.csv', encoding='CP949')
test=pd.read_csv('C:/Users/전수림/Desktop/데이터분석/와인품질분류/data/test.csv', encoding='CP949')
train.head()

train.head()

train.info() #test데이터 + quality컬럼
test.info() # 1000개씩 뽑은 것

###EDA

sns.countplot(x=train['quality']) #6등급의 와인이 가장 많음

# +
#1. 상관관계 파악
# -

plt.figure(figsize=(6, 4))#figure: 그림전체(객체), axes: 내부의 좌표축
sns.heatmap(data=train.corr(), annot_kws={"size":5}, annot=True, fmt='.2f',linewidths=.5, cmap='Reds') 
#heatmap기본문법
# sns.heatmap(data=train.corr(),
#            vmin=100, #최솟값(100이하는 모두 같은 색)
#             vmax=700, #최댓값
#            cbar=False, #colorbar
#             linewidths=.5, #각 셀사이마다 해당 굵기의 선을 넣기
#             annot=True, #annot: 셀안에 숫자출력, fmt=자릿수 출력
#             fmt='.2f',
#             cmap='Reds' #색지정
#            )

plt.figure(figsize=(12,12))
sns.heatmap(data=test.corr(), annot=True, fmt='.2f', linewidths=.5, cmap='Blues')

# +
#각 변수별 분포 파악
plt.figure(figsize=(12,12)) #그래프 사이즈 조정

for i in range(1,13):
#     plt.subplot(3,4,i) #여러개 그래프를 하나의 그림으로 나타냄 plt.subplot(row,col,index)
    sns.displot(train.iloc[:,i]) #히스토그램(train의 i열)
# plt.tight_layout() #subplot간의 간격 자동으로 조절
plt.show()

# -

plt.figure(figsize=(12,12))
for i in range(1,12):
#     plt.subplot(3,4,i)
    sns.displot(test.iloc[:,i])
plt.tight_layout()
plt.show()

# +
#각 독립변수 별로 종속변수의 분포 확인(barplot)

for i in range(11): #11개의 객체 생성
    fig=plt.figure(figsize=(12,6))
    sns.barplot(x='quality',y=train.columns[i+2], data=train)
# -


#object형 제외 모든 컬럼을 StandardScaler
ss=StandardScaler()
numerical_columns = train.select_dtypes(exclude='object').columns.tolist()
train[numerical_columns]=ss.fit_transform(train[numerical_columns])


#전처리 및 모델링
import lightgbm as lgbm # gbm: gradient boosting model의 성능을 높인 모델
sub=pd.read_csv('C:/Users/전수림/Desktop/데이터분석/와인품질분류/data/sample_submission.csv', encoding='CP949')
sub.head()

# #범주형 데이터(red, white를 0,1로 변환하기)
# #1)방법:map으로 직접 지정해주기
# train['type'] = train['type'].map({'white':0, 'red':1}).astype(int)
# test['type'] = test['type'].map({'white':0, 'red':1}).astype(int)
# train.head()
# #2) LabelEncoder()
# from sklearn.preprocessing import LabelEncoder
# encoder=LabelEncoder()
# train['type']=encoder.fit_transform(train['type'].values)
# train.head()
# #3)pd.factorize: 2개의 값 반환(=인코딩된 값, 인코딩된 범주)
# encoded,y_class=pd.factorize(train['type'])
# encoded  #[0,1,0,1,,...]
# y_class #[white,red....]
# train['type']=encoded
# #4)get_dummies
# pd.get_dummies(train['type']) #행렬 반환=>열이름: 범주(red,white) 값=true,false느낌으로 반환


# 독립변수, 종속변수 설정
train_x=train.drop(['index','quality'], axis=1) #axis=1 열값 기준  설정!
train_y=train['quality']
test_x=test.drop('index',axis=1)

train_x.head()

# +
##모델 생성 및 훈련
# -

model=lgbm.LGBMClassifier()
model.fit(train_x, train_y)

y_predict=model.predict(test_x)

y_predict

#예측 파일에 예측값 대입
sub['quality']=y_predict

sub

sub.to_csv('C:/Users/전수림/Desktop/데이터분석/와인품질분류/data/sub.csv',index=False)


