The objective of this project was to implement medical image forgery detection for cloud-based smart healthcare framework by an article published in IEEE. 
# Background:
The Smart healthcare framework proposed is expected to allow patients consult with their doctors without visiting the hospital in person. This will require that several medical information is shared over the cloud. While the proposed framework  provides seamless and real-time healthcare, there are some security challenges that need to be addressed.For instance if a patients mammogram is hacked and a hacker uses copy-move forgery to enlarge
the area of cancer, the diagnosis will be wrong, and the patient will be in life-threatening trouble. Therefore it is important to have systems in place the smart health care framework to check authenticity of information shared over the cloud.

# The proposed smart healthcare framework consist of three main components;
# Clients : 
Patients and doctors/caregivers who interact via web apps or mobile apps. Smart devices or IOTs are responsible for capturing images or data from the patients and uploading them to the cloud.
# Edge computing: 
Designed to provide low-latency and real-time transmission of data between patient and doctors 
# Core cloud:
Which hosts storage devices, virtual machines (servers), a registration and verification unit for the first time and already existing users. The cloud also hosts the image forgery detection module which the project focused on.

# Implementation: 
The main steps involved in the implementation of this work are:
Decomposition of color images into red, green and blue channels. This was done with Opencv
Wiener-filtering of each component of the color image or monochrome image. The output is an image component free from noise
The noise free image is subtracted from the original image to get an estimated noise pattern of the image.The noise pattern is considered as the fingerprint of the image. If any forgery is done, this fingerprint is distorted
Multi-resolution regression filtering is then applied to the noise to capture the relationship between pixel intensities in the image
The output of the filter is fed to two classifiers: the SVM classifier and the ELM classifier. We investigated different kernels of the SVM: linear, polynomial, and radial basis function (RBF).
The scores of the SVM and the ELM are fused via voting
