import cv2
import time
import os
import numpy 
import csv
import sys
from PIL import Image


def useopencv(testcsv_file,train_i):
  csv_file=open(testcsv_file)
  data=csv.reader(csv_file) 
  count_incorrect_fisher=0.0
  count_incorrect_eigen=0.0
  count_incorrect_lbph=0.0
  total_num_face=0
  fisherrecognizer = cv2.createFisherFaceRecognizer()
  lbphrecognizer = cv2.createLBPHFaceRecognizer()
  eigenrecognizer = cv2.createEigenFaceRecognizer()

  eigenrecognizer.load("./eigen_train_num_%s.xml"%(str(train_i)))

  lbphrecognizer.load("./LBPH_train_num_%s.xml"%(str(train_i)))

  fisherrecognizer.load("./fisher_train_num_%s.xml"%(str(train_i)))

  for csvdata2,csvlabel2 in data:
   #    print csvdata
   #    print csvlabel
     total_num_face=total_num_face+1
     f=csvdata2
     l=int(csvlabel2)
     predict_image_pil = Image.open(f).convert('L')
     predict_image = numpy.array(predict_image_pil, 'uint8')
     nbr_predicted_fisher, conf2_fisher = fisherrecognizer.predict(predict_image[:])
     nbr_predicted_eigen, conf2_eigen = eigenrecognizer.predict(predict_image[:])
     nbr_predicted_lbph, conf2_lbph = lbphrecognizer.predict(predict_image[:])
     nbr_actual = l
     if nbr_actual != nbr_predicted_fisher:
            #print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
     #else:
            print "fisherrecognizer {} Picture no.{} is Incorrect Recognized as {}".format(nbr_actual,total_num_face, nbr_predicted_fisher)
            count_incorrect_fisher=count_incorrect_fisher+1.0
     
     if nbr_actual != nbr_predicted_eigen:

            print "eigenrecognizer {} Picture no.{} is Incorrect Recognized as {}".format(nbr_actual,total_num_face, nbr_predicted_eigen)
            count_incorrect_eigen=count_incorrect_eigen+1.0
     
     if nbr_actual != nbr_predicted_lbph:

            print "lbphrecognizer {} Picture no.{} is Incorrect Recognized as {}".format(nbr_actual,total_num_face, nbr_predicted_lbph)
            count_incorrect_lbph=count_incorrect_lbph+1.0

  percentage_fisher=100-(count_incorrect_fisher/total_num_face)*100
  percentage_lbph=100-(count_incorrect_lbph/total_num_face)*100
  percentage_eigen=100-(count_incorrect_eigen/total_num_face)*100

  print "correct rate {} % in fisherrecognizer with {} trainning face".format(percentage_fisher,train_i)

  print "correct rate {} % in lbphrecognizer with {} trainning face".format(percentage_lbph,train_i)

  print "correct rate {} % in eigenrecognizer with {} trainning face".format(percentage_eigen,train_i)


if __name__ == '__main__':
     f_handler=open('OPENCV_recog_rate_jaffe.log', 'w')
     sys.stdout=f_handler
     for train_loop in range(14,19):
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('face image trainning number%i'%int(train_loop))
        useopencv(testcsv_file='jaffe_test.csv',train_i=train_loop)



