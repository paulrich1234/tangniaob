import pandas as pd


data=pd.read_excel('origin_data.xlsx')
label =data['糖尿病']

j=0
for i in label:
    if i ==1:
        j+=1
print(j)