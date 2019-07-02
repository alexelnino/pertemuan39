import numpy as np
import pandas as pd

data=[
    {'luas':1000, 'harga':1000, 'kota':'bekasi' },
    {'luas':2000, 'harga':2000, 'kota':'bekasi' },
    {'luas':3000, 'harga':3000, 'kota':'bekasi' },
    {'luas':1000, 'harga':2000, 'kota':'depok' },
    {'luas':2000, 'harga':4000, 'kota':'depok' },
    {'luas':3000, 'harga':6000, 'kota':'depok' },
    {'luas':1000, 'harga':5000, 'kota':'jakarta' },
    {'luas':2000, 'harga':10000, 'kota':'jakarta' },
    {'luas':3000, 'harga':15000, 'kota':'jakarta' }
]

# # 1. pandas get dummies
# df=pd.DataFrame(data)
# print(df.head(5))
# dfNew=pd.get_dummies(df['kota'])
# # dibuat dummies biar kolom kota diconvert menjadi angka agar bisa diolah
# print(dfNew)
# dfKomplit=pd.concat([df,dfNew], axis='columns')
# dfKomplit=dfKomplit.drop(['kota'], axis='columns')      #menghilangkan kolom kota
# print(dfKomplit.head(5))

# 2. sklearn one hot encoding
df = pd.DataFrame(data)
# 2a. labelling
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['kota'] = label.fit_transform(df['kota'])
# print(df)

dfX=df[['kota', 'luas']].values
# print(dfX)
dfY=df['harga'].values
# print(dfY)

# 2b. one hot encoder       => similar with pandas get dummies
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[0])],        #=>[0]= kolom 'kota' yang mau diubah
    remainder='passthrough'
)

dfX=np.array(coltrans.fit_transform(dfX), dtype=np.int64)
print(dfX)

# linear regression
from sklearn.linear_model import LinearRegression
modelLR = LinearRegression()
modelLR.fit(dfX, dfY)
print(modelLR.predict([[1,0,0,2000]]))