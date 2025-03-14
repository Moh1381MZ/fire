#ketabkhane hay mored niaz
import cv2 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from sklearn.neighbors import KNeighborsClassifier 
from joblib import dump 
data_list=[]
labels=[]
# har tasvir ra shoru be khandan mikonim 
for i,adress in enumerate(glob.glob("fire_dataset\\*\\*")): #ba glob harv tasver ra mi khanim va dar edame shuru be pish pardazesh mikonim

    img=cv2.imread(adress)#khandan tasavir
    img=cv2.resize(img,(500,500))# resize kardan
    img=img/255 #آوردن تصاویر به بازه ی صفر و یک 
    img=img.flatten()#ساخت بردار دو بعدی 
    data_list.append(img)
    #شمارش گر برای شروع پردازش زیبایی کار 
    if i  % 100==0:
        print("[INFO]{}/{} procced".format(i,1000))
    
    #خواندن لیبل ها 
    label=adress.split("\\")[2].split(".")[0]
    labels.append(label)

# کتابخانه اسکی لرن فقط ماتریس یا دیتافریم قبول می کند 
data=np.array(data_list)
#تقسیم بندی تراین و تست
train_x , test_x , train_y,test_y=train_test_split(data,labels,random_state=42,test_size=0.2)
#فایر کردن الگوریتم کی ان ان 

clf=KNeighborsClassifier()
clf.fit(train_x,train_y)
#ذخیره مدل
dump(clf,"fire.z")
acc=clf.score(test_x,test_y)
print(acc)