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

'''
5. 데이터 전처리, 추가, 삭제, 변환
6. gropuby와 pivot table
7. 연결과 병합
'''

'''
5. 데이터 전처리, 추가, 삭제, 변환
'''

# +
from IPython.display import Image
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

# warning 무시
warnings.filterwarnings('ignore')

# e notation 표현 방식 변경
pd.options.display.float_format = '{:.2f}'.format

# 모든 컬럼 표시
pd.set_option('display.max_columns', None)
# -

df = sns.load_dataset('titanic')
df.head()

df1=df.copy()
df1.head()

'''
pandas에서도 insert가 가능함 
RICH라는 새로운 컬럼을 만들어서 값을 insert하기
5번째 위치에 RICH라는 컬럼값을 만들기
'''
df1.insert(5,'RICH',df1['fare']>100) 
df1.head()

'''
행 삭제: drop(n), drop(np.arange(n)), drop([1,3,5,7,9])
'''
# df1.drop(1) 
# df1.drop(np.arange(10))
df1.drop([1,3,5,7,9])

'''
열 삭제 : axis=1 지정!

drop한 결과는 반영하기
1. df1=df1.drop('class', axis=1).head()
2. inplace=True로 반영
''' 
df1=df.copy()
df1.drop('class', axis=1, inplace=True)

df1=df1.drop(['who','deck','alive'], axis=1)

df1

df1=df.copy()
df1['family']=df1['sibsp']+df1['parch']  #가족=[형제+배우자수]+[부모+자녀 수]
# df1.insert(4,'family', df1['sibsp']+df1['parch'])

df1=df.copy()
df1.insert(4,'family', df1['sibsp']+df1['parch'])

#문자열 붙이기
df1['gender']=df1['who']+'-'+df1['sex']

'''
category type으로 변환하기 - 변수에 할당해줘야함!!
catergory 타입은 .cat으로 attribute에 접근할 수 있음

'''
df1['who']=df1['who'].astype('category')
df1['who'].cat.categories=['아이','남자','여자']
df1['who'].value_counts()

# +
'''
서울시 자전거 대여 공공데이터 사용
'''

# !pip install opendata-kr -q

from opendata import dataset

dataset.download('서울시자전거')
# -

bicycle=pd.read_csv('data/seoul_bicycle.csv')
bicycle.head()

bicycle.info()

'''
대여일자가 object로 되어있음 ->datetype으로 변환
1. 변환방법: pd.to_datetime(컬럼.. )
2. date type은 dt접근자로 사용해서 속성에 접근할 수 있음
'''
bicycle['대여일자']=pd.to_datetime(bicycle['대여일자'])

bicycle.info()

bicycle['year']=bicycle['대여일자'].dt.year
bicycle['day']=bicycle['대여일자'].dt.day
bicycle['month']=bicycle['대여일자'].dt.month
bicycle['week']=bicycle['대여일자'].dt.weekday

bicycle.head()

'''
pd.cut():구간나누기(수치형을 구간으로 나눠 카테고리화 할때 사용)
pd.cut(데이터, 구간개수(혹은 구간),right=T/F, labels=labels) 
right=False: 큰쪽 범위를 포함하지 않음(=작은쪽 범위 포함). ex)1000000<=많음<max값
right=True: 큰쪽 범위를 포함 ex)1000000<많음<=max값
labels=labels(변수)
'''
bins=[0,6000,100000, bicycle['이동거리'].max()]
labels=['적음','보통','많음']
bicycle['이동거리_cut']=pd.cut(bicycle['이동거리'], bins, labels=labels, right=False)
#혹은
bicycle['이동거리_cut']=pd.cut(bicycle['이동거리'], bins=3)
bicycle['이동거리_cut'].value_counts() #이렇게 임의로 3구간을 나누면 데이터쏠림 현상이 발생할 수 있음 =>qcut()을 사용


bicycle['이동거리_qcut']=pd.qcut(bicycle['이동거리'], q=3) 
bicycle['이동거리_qcut'].value_counts() 
#3구간이 고르게 분포할 수 있도록 나눠주지만, 범위의 간격이 일정하지 않음 =>임의범위를 조정할 수 있음

qcut_bins=[0,0.2,0.8,1]
qcut_labels=['많음','보통','적음']
bicycle['이동거리_qcut2']=pd.qcut(bicycle['이동거리'], qcut_bins, labels=qcut_labels) 
bicycle['이동거리_qcut2'].value_counts() 

df1=df.copy()
df1.head()

df1.drop([1,3,5], inplace=True)

df1.drop(['embarked','class','alone'], axis=1, inplace=True)

df1.head(10)

'''
species: 붓꽃 데이터의 종류

sepal_length: 꽃받침의 길이

sepal_width: 꽃받침의 넓이

petal_length: 꽃잎의 길이

petal_width: 꽃잎의 넓이
'''
iris=sns.load_dataset('iris')
iris.head()

iris['sepal']=iris['sepal_length']*iris['sepal_width']

iris.head()

iris['petal']=iris['petal_length']*iris['petal_width']
iris.head()

iris.drop(['petal_length','petal_width'], axis=1, inplace=True)

iris.head()

setosa=iris.loc[(iris['species']=='setosa')]
setosa.sort_values(by='sepal', ascending=False).head(10)

iris['sepal'].mean()-iris['petal'].mean()

df2 = pd.read_csv('data/seoul_bicycle.csv')
df2.head()

df2.info()

df2['대여일자']=pd.to_datetime(df2['대여일자'])

df2['연도']=df2['대여일자'].dt.year
df2['월']=df2['대여일자'].dt.month
df2['일']=df2['대여일자'].dt.day
df2['요일']=df2['대여일자'].dt.dayofweek
df2.head()

sample = sns.load_dataset('titanic')
sample.head(3)

'''
pd.cut():구간나누기(수치형을 구간으로 나눠 카테고리화 할때 사용)
pd.cut(데이터, 구간개수(혹은 구간),right=T/F, labels=labels) 
right=False: 큰쪽 범위를 포함하지 않음(=작은쪽 범위 포함). ex)1000000<=많음<max값
right=True: 큰쪽 범위를 포함 ex)1000000<많음<=max값
labels=labels(변수)
'''
age_bin=[0,15,30,45,max(sample['age'])]
pd.cut(sample['age'],age_bin, right=True)

label=['young','normal','old']
pd.qcut(sample['age'], q=3, labels=label).value_counts()

'''
6. groupby와 pivot table 
groupby: 데이터를 그룹화해서 통계량을 봄
pivot_table(): 데이터를 특정 조건에 따라 행렬을 기준으로 데이터를 펼쳐서 봄
'''

df = sns.load_dataset('titanic')
df.head()
df.info()

'''
apply: 함수를 적용
데이터['컬럼'].apply(함수)
'''


def transform_who(x):
    return x['fare'] / x['age']
df.apply(transform_who, axis=1)
#axis=1 열을 기준으로 함수를 적용하는 것 잊지 않기..!! ->열값을 뽑아와서 계산하는 것이기 때문! 

# #lambda함수로 apply 적용해보기
df['survived'].apply(lambda x: '생존' if x==1 else '사망').value_counts()

'''
groupby()를 사용할 때는 aggregate하는 통계함수와 일반적으로 같이 적용
'''
df.groupby('sex').mean()

df.groupby(['sex','pclass']).mean()

#모든 컬럼에 대한 값이 아닌, 하나의 열만 뽑아서 보고 싶다면
df.groupby(['sex','pclass'])['survived'].mean()

#데이터프레임으로 출력
#1번째
pd.DataFrame(df.groupby(['sex','pclass'])['survived'].mean())
#2번쨰: survived열을 2차원행렬로 출력
df.groupby(['sex','pclass'])[['survived']].mean()

#응용: 그룹핑된 데이터프레임의 index를 초기화
df.groupby(['sex','pclass'])[['survived']].mean().reset_index()
#agg도 사용가능
df.groupby(['sex','pclass'])[['survived','age']].agg(['mean','sum'])

'''gropuby연습문제'''
sample = df.copy()
sample.head()


# +

#class 컬럼의 값을 다음과 같이 바꾸고, 분포를 출력후 변경 전과 동일한지 확인하세요

def change(x):
    if x=='Third':
        return '삼등석'
    elif x=='First':
        return '일등석'
    else:
        return '이등석'
        
sample['class'].apply(change).value_counts()
# -

#1.pclass 별 생존율
sample.groupby('pclass')['survived'].mean()

#2.embarked 별 생존율 통합 통계
sample.groupby('embarked')['survived'].agg(['mean', 'var'])

#who, pclass별 생존율, 생존자수
sample.groupby(['who','pclass'])['survived'].agg(['mean','sum'])

'''
이렇게 구현하면 여자,남자 따로 반영하기 어려움
woman=(sample['who']=='woman')
woman_mean=sample.loc[woman,'age'].mean()
sample.loc[woman,'age'].fillna(woman_mean)
--> groupby로 성별로 age통계값을 구하고 적용(apply)시킬 수 있음
'''
sample['age']=sample.groupby('sex')['age'].apply(lambda x: x.fillna(x.mean()))

print(sample['age'].isnull().sum())
print(f"age 평균: {sample['age'].mean():.2f}")

'''
pivot_table: groupby와 유사하지만, 행렬로 계산함(엑셀의 pivot과 유사)
groupy는 '이 열을 기준으로 통계를 보여줘!'라는 느낌 vs pivot_table은 행렬을 내가 지정해서 정보를 볼 수 있음
index, columns, values를 지정해서 피벗함
'''
#인덱스(행)에 who별로 survived값들을 보여줘
df.pivot_table(index='who', values='survived')

df.pivot_table(columns='who', values='survived')

#행렬별 값을 정리
df.pivot_table(index='who', columns='pclass', values='survived')

#다중 통계함수 적용
df.pivot_table(index='who',columns='pclass',values='survived',aggfunc=['sum','mean'])

tips = sns.load_dataset('tips')
tips.head()

#tip에 대한 평균값을 산출합니다.
tips.pivot_table(index='smoker', columns='day', values='tip', aggfunc='mean')

tips.pivot_table(index='day', columns='time', values='total_bill', aggfunc=['mean','sum'])

# +
# !pip install opendata-kr -q

from opendata import dataset

# 유가정보 데이터 다운로드
dataset.download('유가정보')
# -

gas1 = pd.read_csv('data/gas_first_2019.csv', encoding='euc-kr')
gas1.head()
#1월 부터 6월 까지 상반기 데이터 로드


gas2 = pd.read_csv('data/gas_second_2019.csv', encoding='euc-kr')
gas2.head()

'''
concat() : dataframe을 행/열방향으로 연결(다른 데이터프레임2개를 연결하는 느낌)
merge(): dataframe을 특정 key를 기준으로 병합(같은 키값을 가지고 있고, 열을 병합하는 느낌)
'''
gas=pd.concat([gas1,gas2],ignore_index=True)
#인덱스가 초기화되지 않으면 전체 df의 개수와 index개수가 맞지 않음
gas

gas11 = gas1[['지역', '주소', '상호', '상표', '휘발유']]
gas22 = gas2[['상표', '번호', '지역', '상호', '주소', '경유', '휘발유']]

#없는 컬럼은 nan으로 concat되었음
pd.concat([gas11, gas22], ignore_index=True)

#열 연결
gas1 = gas.iloc[:, :5]
gas2 = gas.iloc[:, 5:]
pd.concat([gas1,gas2],axis=1)

'''
merge()
다른 dataframe이지만, 같은 key값(id or name 등)이 있다면 merge가능
'''
df1 = pd.DataFrame({
    '고객명': ['박세리', '이대호', '손흥민', '김연아', '마이클조던'],
    '생년월일': ['1980-01-02', '1982-02-22', '1993-06-12', '1988-10-16', '1970-03-03'],
    '성별': ['여자', '남자', '남자', '여자', '남자']})
df2 = pd.DataFrame({
    '고객명': ['김연아', '박세리', '손흥민', '이대호', '타이거우즈'],
    '연봉': ['2000원', '3000원', '1500원', '2500원', '3500원']})


pd.merge(df1,df2)

#병합 옵션 4가지 how='left' right, outer, inner: sql의 join과 같은 속성!
pd.merge(df1, df2, how='left')
pd.merge(df1, df2, how='outer')

df1 = pd.DataFrame({
    '이름': ['박세리', '이대호', '손흥민', '김연아', '마이클조던'],
    '생년월일': ['1980-01-02', '1982-02-22', '1993-06-12', '1988-10-16', '1970-03-03'],
    '성별': ['여자', '남자', '남자', '여자', '남자']})
df2 = pd.DataFrame({
    '고객명': ['김연아', '박세리', '손흥민', '이대호', '타이거우즈'],
    '연봉': ['2000원', '3000원', '1500원', '2500원', '3500원']})

#병합하려는 컬럼의 이름이 다른 경우, left_on과 right_on을 모두 지정
pd.merge(df1,df2, left_on='이름',right_on='고객명')


