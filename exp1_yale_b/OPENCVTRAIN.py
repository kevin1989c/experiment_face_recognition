import cv2
import time
import os
import numpy 
import csv
from PIL import Image
# For face detection we will use the Haar Cascade provided by OpenCV.
#cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)



def get_images_and_labels(data_csvfile,img_h,img_w,total_image_num,):

    csv_file=open(data_csvfile)
    data=csv.reader(csv_file)

    images = numpy.empty(((total_image_num,img_h,img_w)),dtype=int)
  
    labels = numpy.empty(total_image_num,dtype=int)
    i_1=0
    for csvdata,csvlabel in data:
        print csvdata
        print csvlabel
        f=csvdata
        l=int(csvlabel)

        image_pil = Image.open(f).convert('L')
        image = numpy.array(image_pil, 'uint8')
         
        images[i_1]=image[:]
        labels[i_1]=l
        i_1=i_1+1
        #images.append(image[:])
        #labels.append(nbr)

    return images, labels

def train_opencv(train_i,img_h,img_w,total_image_num,person_num,datacsv_file):
    lbphrecognizer = cv2.createLBPHFaceRecognizer()
    eigenrecognizer =  cv2.createEigenFaceRecognizer()
    fisherrecognizer =  cv2.createFisherFaceRecognizer()
    
    images, labels = get_images_and_labels(data_csvfile=datacsv_file,img_h=img_h,img_w=img_w,total_image_num=total_image_num)


    train_data = numpy.empty(((train_i*person_num,img_h,img_w)),dtype=int)
    train_label = numpy.empty(train_i*person_num,dtype=int)
    num_set=total_image_num/person_num
    

    for s in range(person_num):

       train_data[s*train_i:s*train_i+train_i]=images[s*num_set:s*num_set+train_i]
       train_label[s*train_i:s*train_i+train_i]=labels[s*num_set:s*num_set+train_i]


    lbphrecognizer.train(train_data, numpy.array(train_label))
    lbphrecognizer.save("./LBPH_train_num_%s.xml"%(str(train_i)))
    fisherrecognizer.train(train_data, numpy.array(train_label))
    fisherrecognizer.save("./fisher_train_num_%s.xml"%(str(train_i)))
    eigenrecognizer.train(train_data, numpy.array(train_label))
    eigenrecognizer.save("./eigen_train_num_%s.xml"%(str(train_i)))
   


if __name__ == '__main__':
     for train_loop in range(10,29):
        print('face image trainning number%i'%int(train_loop))
        train_opencv(img_h=58,img_w=50,total_image_num=784,person_num=28,datacsv_file='yale_b_train.csv',train_i=train_loop)


