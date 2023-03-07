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
실습파일

https://wikidocs.net/book/4639

1.파이썬 기본지식
2.조회, 정렬,조건
3.통계
4.복사와 결측치
'''


from IPython.display import Image
import pandas as pd
import numpy as np
import seaborn as sns

df=sns.load_dataset('titanic')
df.head()

'''
조회, 정렬, 조건
'''
df.info()

#pclass는 male,female,child로 이뤄져있어서 categrory화 
df['pclass'].astype('category').head()


#정렬: 기본-내림차순으로 정렬, 기준컬럼 지정가능
df.sort_values(by='age',ascending=False).head()

#여러가지 컬럼으로 정렬하기: 앞쪽 컬럼이 먼저 정렬됨..
df.sort_values(by=['age','fare'], ascending=False).head()

#loc[행,렬]: 위치값(label)으로 찾기, iloc: 인덱스로 찾기
df.loc[5,'class']

# +
'''
주의! loc안에 loc는 쓰지 않음->열값으로 해당 조건 설정
주의! 조건 줄 때 and,or이 아닌 & | 로 조건 지정 and는 논리연산자, &는 비트연산자
주의! : 각 조건은 ()로 감싸줘야함
'''
df.loc[(df['class']=='First') | (df['fare']>70), 'age']

#위는 복잡하니, 변수에 저장하는게 더 좋을 듯!
con1=(df['class']=='First')
con2=(df['fare']>70)
df.loc[con1|con2, 'age']
# -

#인덱싱 연산
df.iloc[[2,3,4], 2:4]

'''
isin: 특정값의 포함여부를 알아보는 것(파이썬의 in은 사용불가능하고, 대신 isin사용)
'''
sample=pd.DataFrame({'name':['kim','lee','park','choi'],
                     'age': [24,27,34,19]
                    })
#sample이름이 kim,lee가 있는지(조건)과 맞는 데이터를 찾아옴
sample.loc[sample['name'].isin(['kim','lee'])]

#탑승항구별 승객 데이터 분포: 함수형태로 써야함을 주의!
df['embark_town'].value_counts()

df['who'].value_counts()

#tip: 레스토랑의 매출과 팁을 나타내는 데이터
tips=sns.load_dataset('tips')
tips.head()

# +
'''
수정된 내용을 적용할 땐 데이터프레임형식의 변수에 다시 저장해줘야 함
'''

df2=pd.DataFrame() #빈 데이터프레임 만들기
df2=tips.sort_values(by=['total_bill','tip'],ascending=False).head(10)
# tips.sort_values(by=['total_bill','tip'],ascending=False).head(10)
df2
# -

tips.loc[((tips['day']=='Fri')| (tips['day']=='Sat')) & (tips['tip']<10), ['total_bill','tip','smoker','time']].head(10)

df[3:7]

#그냥 df[2:['age','who']]로 하면 오류뜸
df.loc[2:10:2, ['age','who']] 

# 나이가 30살 이상 남자 승객 조건 필터링
# fare를 많이 낸 순서로 내림차순 정렬
# 상위 10개를 출력
df.loc[df['age']>=30].sort_values(by='fare',ascending=False).head(10)

# 나이가 20살 이상 40살 미만인 승객
# pclass가 1등급 혹은 2등급인 승객
# 열(column)은 survived, pclass, age, fare 만 나오게 출력
# 10개만 출력
df.loc[((df['age']>=20) & (df['age']<40)) & ((df['pclass']==1) | (df['pclass']==2)),['survived','pclass','age','fare']].head(10)

'''
---------------------------------------------------------------
통계
'''
df.describe().T

#문자열 컬럼의 통계값
#count, unique, top:가장 많이 출현한 데이터, freq: 가장 많이 출현한 데이터의 빈도수
df.describe(include='object')

'''
skipna: na를 뛰어넘을건지, skipna=False는 nan값이 있는 column은 nan로 출력
'''
df.mean(skipna=False)

#median:중앙값: 오름차순 정렬 후 가운데값, 짝수개이면, 중앙데이터의 평균값을 반환
pd.Series([1,2,3,4,5,6]).median()

'''
누적합 cumsum() / cumprod(): 누적곱
'''
df['age'].cumsum()

#분산, 표준편차
df['fare'].var()
np.sqrt(df['fare'].var())

'''
agg: aggregation: 복수의 통계함수 적용 like map
데이터.agg([연산자1, 연산자2... ])
'''
df[['age','fare']].agg(['min','max','count','mean'])

#고유값의 개수
df['who'].nunique()

#상관관계 확인(특정 컬럼을 반영하려면 뒤에 []사용)
df.corr()['survived'] #parch와 가장 높은 상관관계

#실습---------------------------------
df2=pd.DataFrame()
df2=df.loc[(df['fare']>=30) & (df['fare']<40) & (df['pclass']==1)]

df2['age'].count()

df2['age'].mean()

diamond = sns.load_dataset('diamonds')
diamond

diamond['depth'].min()

#carat에 대한 평균과 분산을 동시에 출력하세요.. -> 동시에 연산자를 적용하는 함수...agg!
diamond['carat'].agg(['mean','var'])

diamond[['x','y']].agg(['sum','std'])

penguin = sns.load_dataset('penguins')
penguin

penguin['species'].unique()

penguin['island'].mode()

penguin['body_mass_g'].quantile(0.1)

penguin['body_mass_g'].quantile(0.8)

'''
복사와 결측치-따로 데이터프레임 만들지 않고 복사본을 만들어서 사용하기
'''
df_copy=df.copy()
df_copy.head()

#결측치
df.isnull().sum()
df.isna().sum()

df.notnull().sum()

df.loc[df['age'].isnull()].head()

#fillna의 값을 저장하려면 변수에 저장해줘야한다
df_copy['age']=df_copy['age'].fillna(df['age'].mean())
df_copy.loc[df_copy['age'].isnull()].head()

#최빈값(mode)로 채울 때에는 반드시 0번째 index지정하여 값을 추가해야함
df_copy=df.copy()
df_copy['deck'].fillna(df_copy['deck'].mode()[0]).tail()

'''
연습문제 age가 결측치인 승객의 나이를 30세
'''
df_copy['age']=df_copy['age'].fillna(30)
assert df['age'].isnull().sum() == 0

'''
내풀이
df_copy2=df.copy()
df_copy3=df.copy()
df_copy2=df.loc[df['sex']=='male']
df_copy3=df.loc[df['sex']=='female']
df_copy2['age']=df_copy2['age'].fillna(df_copy2['age'].mean())
df_copy3['age']=df_copy3['age'].fillna(df_copy3['age'].mean())

데이터프레임1개, 조건을 세분화하는 방법
'''
df_copy2=df.copy()
female=(df_copy2['sex']=='female')
male=(df_copy2['sex']=='male')
female_age=df_copy2.loc[female,'age'].mean()
male_age=df_copy2.loc[male,'age'].mean()
df_copy2.loc[female,'age']=df_copy2.loc[female,'age'].fillna(female_age)
df_copy2.loc[male,'age']=df_copy2.loc[male,'age'].fillna(male_age)


assert (df_copy2['age'].isnull().sum() == 0)
assert df_copy2['age'].mean().round(5) == 29.73603
