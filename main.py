import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
from PIL import ImageChops, Image
import cv2
from skimage import color, filters


save_dir = 'dataset'
Au_pic_list = glob('Au' + os.sep + '*')
Sp_pic_list = glob('Sp' + os.sep + '*')
au_index = [6, 9, 10, 14]
background_index = [14, 21]
foreground_index = [22, 29]

def find_background(Au_pic_list, Sp_pic_list):
    ### find spliced images with Au_pic as background
    # Au_pic: the path of an authentic image
    # Sp_pic_list: all paths of spliced images
    # result(return): a list of paths with Au_pic as background
    backgrounds = []
    Au_images = []
    
    for Au_pic in Au_pic_list:
        au_name = Au_pic[au_index[0]:au_index[1]] + Au_pic[au_index[2]:au_index[3]]
    
        for spliced in Sp_pic_list:
            sp_name = spliced[background_index[0]:background_index[1]]
            if au_name == sp_name:
                #print(f"sp = {sp_name}, au = {au_name}")
                if spliced not in backgrounds:
                    backgrounds.append(spliced)
                    
                if Au_pic not in Au_images:
                    Au_images.append(Au_pic)
                    #print(f"Spliced = {spliced}")

    return Au_images, backgrounds

def splice_save(au, backgrounds, save_dir):
    # splice together Au_pic and each of backgrounds, and save it/them.
    # Au_pic: the path of an authentic image
    # backgrounds: list returned by `find_background`
    # save_dir: path to save the combined image
    for Au_pic in au:
        au_image = plt.imread(Au_pic)
        for each in backgrounds:
            sp_image = plt.imread(each)
            if au_image.shape == sp_image.shape:
                result = np.concatenate((au_image, sp_image), 1)
                plt.imsave(save_dir+os.sep+each[14:], result)


#for Au_pic in Au_pic_list[0]:
au_list, sp_list= find_background(Au_pic_list, Sp_pic_list)
    #splice_save(Au_pic, backgrounds, save_dir)
 #%%
#File Path
filepath_Au = '/Users/user/Desktop/North Carolina A&T State University/Digital Signal Processing/Project/casia-dataset/CASIA1/Au'
filepath_Sp = '/Users/user/Desktop/North Carolina A&T State University/Digital Signal Processing/Project/casia-dataset/CASIA1/Sp'

Au_file_names =[]
Sp_file_names =[]

for file_path in au_list:
    Au_file_names.append(os.path.basename(file_path))
    

for file_path_sp in sp_list:
    Sp_file_names.append(os.path.basename(file_path_sp))


#%%
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def multiResolutionRegressioin(image_noise, wave_type='db5'):
    import pywt

    #titles = ['Approximation', ' Horizontal detail',
    #'Vertical detail', 'Diagonal detail']

#    coeffs2 = pywt.dwt2(image_noise, 'db9')
#    LL, (LH, HL, HH) = coeffs2
    
#    return LH
    
    dftcoeffs = pywt.wavedec(image_noise,wave_type,mode='symmetric',level=9, axis=-1)
    for i in range(9):
        dftcoeffs[-i]=np.zeros_like(dftcoeffs[-i])
    
    filtered_data = pywt.waverec(dftcoeffs,wave_type,mode='symmetric', axis=-1)
    
    return filtered_data
    


def get_noise_after_wiener_normalize(filepath_Au, Au_file_names):
    resArray = []
    
    kernel = gaussian_kernel(4)
    print(kernel)
    for filename in os.listdir(filepath_Au):
        if os.path.basename(filename) in Au_file_names:
            image = cv2.imread(os.path.join(filepath_Au, filename))
            
            if image.ravel().shape == (294912,):
                #print(image.ravel().shape)
                gray_image = color.rgb2grey(image)
        
                ## Apply Wiener Filter
                filtered_img = wiener_filter(gray_image , kernel, K = 5)
        
                ## Get Noise from Image
                noise = cv2.subtract(gray_image,filtered_img)
        
                # Append noise image to result
                mrrImage = multiResolutionRegressioin(noise)
                #mrrImage = multiResolutionRegressioin(noise, wave_type='db5')
                
            
                norm_image = cv2.normalize(mrrImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
                resArray.append(norm_image.ravel())
                #print(norm_image.ravel())
                #print(cnt)
                #print(os.path.basename(filename))
                            
    return np.array(resArray)    
#%% Putting data into an array and adding lable

XresultsArray_Au = get_noise_after_wiener_normalize(filepath_Au, Au_file_names)
XresultsArray_Sp = get_noise_after_wiener_normalize(filepath_Sp,Sp_file_names)

y_labels_Au = np.repeat(1,len(XresultsArray_Au))
y_labels_Sp = np.repeat(0,len(XresultsArray_Sp))

#%% SVM
from sklearn.model_selection import train_test_split
from sklearn import svm

features = np.vstack((XresultsArray_Au, XresultsArray_Sp))
labels = np.hstack((y_labels_Au , y_labels_Sp))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=2)

# SVM

model_svm = svm.SVC(kernel='rbf', gamma='auto')
model_svm.fit(X_train, y_train) 
model_svm.predict(X_test)

#Cross Validation  for SVM

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



cross_val_score(model_svm, X_train, y_train, cv=5)
#
#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ELM1 import BinaryClassify

#load data
features = np.vstack((XresultsArray_Au, XresultsArray_Sp))
labels = np.hstack((y_labels_Au , y_labels_Sp))

X = features
y = labels[:,np.newaxis]

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)


# Split data frame
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create classify
elm_model = BinaryClassify(hidden_layer=['relu', 'linear', 'sigmoid'], output_layer='sigmoid') 

input_dim = X.shape[1] # number of features
elm_model.buildNetwork(input_dim)


elm_model.fit(X_train, y_train)

print(elm_model.evaluete(X_test, y_test)[1])
#%% Decison tr

#from sklearn.linear_model import LogisticRegression

#clf1 = LogisticRegression().fit(X_train, y_train)


#cross_val_score(clf1, X_train, y_train, cv=3)

##%%
#from sklearn import tree
#
#clf_desci= tree.DecisionTreeClassifier()
#clf_desci = clf_desci.fit(X_train, y_train)
#
#cross_val_score(clf_desci, X_train, y_train, cv=10)

#%%
#from sklearn.ensemble import VotingClassifier
#from sklearn.model_selection import cross_val_score
#
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=2)
#
#fusedModel = VotingClassifier(estimators=[('svm', model_svm), ('elm', elm_model)], voting='hard')
#fusedModel.fit(X_train, y_train)
#

#cross_val_score(fusedModel, X_train, y_train, cv=5)

#%%

Accuracy = [63.4 ,66.5, 72]

index = ['SVM', 'ELM', 'SVM + ELM']
df = pd.DataFrame({'Accuracy': Accuracy}, index=index)

ax = df.plot.bar(rot=0)



