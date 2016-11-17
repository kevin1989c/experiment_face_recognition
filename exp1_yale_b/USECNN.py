
import os
import sys
import cPickle
import csv
import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv



def load_params(train_i):
    fr=open('params_train_i_%s.pkl'%(str(train_i)),'rb')
    layer0_params=cPickle.load(fr)
    layer1_params=cPickle.load(fr)
    layer2_params=cPickle.load(fr)
    layer3_params=cPickle.load(fr)
    fr.close()
    return layer0_params,layer1_params,layer2_params,layer3_params





def load_data(datacsv_file, total_image_num, img_h, img_w):
  csv_file=open(datacsv_file)
  data=csv.reader(csv_file)
  faces=numpy.zeros((total_image_num,img_h*img_w))   
  i=0
  label=numpy.zeros(total_image_num)
  for csvdata,csvlabel in data:
     #print csvdata
     #print csvlabel
     f=csvdata
     l=int(csvlabel)
     img1=Image.open(f)
     img=img1.convert('L')
     img_ndarray=numpy.asarray(img,dtype='float64')/256
     #print i
     faces[i]=numpy.ndarray.flatten(img_ndarray[:])
     label[i]=l
     i=i+1
  
  return faces,label


class LogisticRegression(object):
    def __init__(self, input, params_W, params_b, n_in, n_out):
        self.W = params_W
        self.b = params_b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, input, params_W,params_b, n_in, n_out,
                 activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

	

class LeNetConvPoolLayer(object):
    def __init__ (self, input, filter_w, filter_h, filter_num, img_w, img_h, input_feature,batch_size, poolsize,params_W,params_b):

        self.input = input
        self.W = params_W
        self.b = params_b

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=(filter_num, input_feature, filter_h, filter_w),
            image_shape=(batch_size,input_feature,img_h,img_w)
        )

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]




def use_CNN(train_i,datacsv_file,filter_num1, filter_num2,filter_h,filter_w,img_h,img_w,poolsize,input_feature,total_image_num,person_num, count_incorrect=0.0): 
    
   
    faces,label=load_data(datacsv_file,total_image_num, img_h, img_w)


  

    layer0_params,layer1_params,layer2_params,layer3_params=load_params(train_i)
    
    x = T.matrix('x')  

    layer0_input = x.reshape((total_image_num, input_feature, img_h, img_w))
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num1, 
        img_h=img_h,         
        img_w=img_w, 
        input_feature=input_feature,
        batch_size=total_image_num, 
        poolsize=poolsize
    )
    layer1_input=layer0.output
    layer1_image_h=(img_h-filter_h+1)/2
    layer1_image_w=(img_w-filter_w+1)/2
    layer1 = LeNetConvPoolLayer(
        input=layer1_input,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num2,  
        img_h=layer1_image_h, 
        img_w=layer1_image_w, 
        input_feature=filter_num1,
        batch_size=total_image_num,  
        poolsize=poolsize
    )

    
    layer2_input = layer1.output.flatten(2)
    layer2_image_h=(layer1_image_h-filter_h+1)/2
    layer2_image_w=(layer1_image_w-filter_w+1)/2

    layer2 = HiddenLayer(
        input=layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        n_in=filter_num2 *  layer2_image_h * layer2_image_w,
        n_out=500,      
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, params_W=layer3_params[0],params_b=layer3_params[1],n_in=500, n_out=person_num)  

 
     

    f = theano.function(
        [x],    
        layer3.y_pred
    )
    

    pred = f(faces)
    


    for i in range(total_image_num): 
	 if pred[i] != label[i]:
                #print('picture: %i is person %i, but mis-predicted as person %i' %(i, label[i], pred[i]))
                count_incorrect=count_incorrect+1.0
    print('face number:  %i'%total_image_num)
    percentage=100-(count_incorrect/total_image_num)*100
    print "correct rate {} % in CNNrecognizer".format(percentage)


if __name__ == '__main__':
      f_handler=open('CNN_recog_rate_yaleB.log', 'w')
      sys.stdout=f_handler
      for train_i in range(10,22):
        print('face image trainning number%i'%int(train_i))
	use_CNN(train_i,
        datacsv_file='yale_b_test.csv',
        filter_num1=6,
        filter_num2=16,
        filter_h=5,
        filter_w=5,
        img_h=58,
        img_w=50,
        poolsize=(2,2),
        total_image_num=364,
        person_num=28,
        input_feature=1)






