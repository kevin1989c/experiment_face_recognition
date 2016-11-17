import os
import sys
import time
import csv
import numpy
from PIL import Image
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import cPickle  


def load_data(datacsv_file, train_i, valid_i, total_image_num, person_num, img_h, img_w):
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

  train_data=numpy.zeros((train_i*person_num,img_h*img_w))
  train_label=numpy.zeros(train_i*person_num)
  valid_data=numpy.zeros((valid_i*person_num,img_h*img_w))
  valid_label=numpy.zeros(valid_i*person_num)
  test_data=numpy.zeros((person_num,img_h*img_w))
  test_label=numpy.zeros(person_num)
  num_set=total_image_num/person_num
  for i in range(person_num):

   train_data[i*train_i:(i*train_i+train_i)]=faces[i*num_set:(i*(num_set)+train_i)]
   train_label[i*train_i:(i*train_i+train_i)]=label[i*(num_set):(i*(num_set)+train_i)]
   valid_data[i*valid_i:(i*valid_i+valid_i)]=faces[(i*(num_set)+train_i):(i*(num_set)+train_i+valid_i)]
   valid_label[i*valid_i:(i*valid_i+valid_i)]=label[(i*(num_set)+train_i):(i*(num_set)+train_i+valid_i)]
   test_data[i]=faces[i*(num_set)+(num_set-1)]
   test_label[i]=label[i*(num_set)+(num_set-1)]
 

  def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')



  train_set_x, train_set_y = shared_dataset(train_data,train_label)
  valid_set_x, valid_set_y = shared_dataset(valid_data,valid_label)
  test_set_x, test_set_y = shared_dataset(test_data,test_label)
  rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
  return rval




class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
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
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    def __init__ (self, rng, input, filter_w, filter_h, filter_num, img_w, img_h, input_feature,batch_size, poolsize):
        
        self.input = input

        

        fan_in = filter_w*filter_h*input_feature
        fan_out = (filter_num*filter_h*filter_w) /numpy.prod(poolsize)

        # initialize weights with random weights
        W_shape = (filter_num, input_feature, filter_w, filter_h)
        W_bound = numpy.sqrt(1. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=W_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_num,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

       
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=(filter_num, input_feature, filter_h, filter_w),
            image_shape=(batch_size,input_feature,img_h,img_w)
        )

       
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]



def save_params(train_i,param1,param2,param3,param4):  
        write_file = open('params_train_i_%s.pkl'%(str(train_i)), 'wb')   
        cPickle.dump(param1, write_file, -1)
        cPickle.dump(param2, write_file, -1)
        cPickle.dump(param3, write_file, -1)
        cPickle.dump(param4, write_file, -1)
        write_file.close()  




def train_cnn(datacsv_file,train_i,valid_i,learning_rate, n_epochs,batch_size,filter_num1, filter_num2,filter_h,filter_w,total_image_num,person_num,img_h,img_w,poolsize,input_feature):   

    rng = numpy.random.RandomState(23455)
    
    datasets = load_data(datacsv_file,train_i,valid_i,total_image_num,person_num,img_h,img_w)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

   
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

   
    index = T.lscalar()
    x = T.matrix('x')  
    y = T.ivector('y')



   
    print '... building the model'


#reshape the input into 4D
    layer0_input = x.reshape((batch_size, input_feature, img_h, img_w))


    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num1, 
        img_h=img_h,         
        img_w=img_w, 
        input_feature=input_feature,
        batch_size=batch_size, 
        poolsize=poolsize
    )
    layer1_input=layer0.output
    layer1_image_h=(img_h-filter_h+1)/2
    layer1_image_w=(img_w-filter_w+1)/2
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer1_input,
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num2,  
        img_h=layer1_image_h, 
        img_w=layer1_image_w, 
        input_feature=filter_num1,
        batch_size=batch_size,  
        poolsize=poolsize
    )
    
    layer2_input = layer1.output.flatten(2)
    layer2_image_h=(layer1_image_h-filter_h+1)/2
    layer2_image_w=(layer1_image_w-filter_w+1)/2

    
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=filter_num2 *  layer2_image_h * layer2_image_w,
        n_out=500,      
        activation=T.tanh
    )

   
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=person_num)  


  
    cost = layer3.negative_log_likelihood(y)
    
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

   
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    grads = T.grad(cost, params)
    
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
   
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    print '... training'
   
    patience = 2000
    patience_increase = 2  
    improvement_threshold = 0.99  
    validation_frequency = min(n_train_batches, patience / 2) 


    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
		    save_params(train_i,layer0.params,layer1.params,layer2.params,layer3.params)

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




if __name__ == '__main__':

     for train_num in range(10,22):
        #train_num=12
        print('---------------------------------------------------------------------------------------------------')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
        print('--trainning face number - %i-images-per person\n'%int(train_num))
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('---------------------------------------------------------------------------------------------------')
	train_cnn(datacsv_file='yale_b_train.csv',
                  train_i=train_num,
                  valid_i=6,
                  learning_rate=0.003,
                  n_epochs=400,
                  batch_size=10,
                  filter_num1=6,
                  filter_num2=16,
                  filter_h=5,
                  filter_w=5,
                  total_image_num=784,
                  person_num=28,
                  img_h=58,
                  img_w=50,
                  poolsize=(2,2),
                  input_feature=1)

















