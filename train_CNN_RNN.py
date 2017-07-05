from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
import numpy as np
import glob,os
from scipy.misc import imread,imresize

batch_size = 128

def bring_data_from_directory():
  datagen = ImageDataGenerator(rescale=1. / 255)
  train_generator = datagen.flow_from_directory(
          'train',
          target_size=(224, 224),
          batch_size=batch_size,
          class_mode='categorical',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['class_1','class_2','class_3','class_4','class_5'])

  validation_generator = datagen.flow_from_directory(
          'validate',
          target_size=(224, 224),
          batch_size=batch_size,
          class_mode='categorical',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['class_1','class_2','class_3','class_4','class_5'])
  return train_generator,validation_generator

def load_VGG16_model():
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  print "Model loaded..!"
  print base_model.summary()
  return base_model

def extract_features_and_store(train_generator,validation_generator,base_model):
  '''
  x_generator = None
  y_lable = None
  batch = 0
  for x,y in train_generator:
      if batch == (56021/batch_size):
          break
      print "predict on batch:",batch
      batch+=1
      if x_generator==None:
         x_generator = base_model.predict_on_batch(x)
         y_lable = y
         print y
      else:
         x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
         y_lable = np.append(y_lable,y,axis=0)
  x_generator,y_lable = shuffle(x_generator,y_lable)
  np.save(open('video_x_VGG16.npy', 'w'),x_generator)
  np.save(open('video_y_VGG16.npy','w'),y_lable)
  batch = 0
  x_generator = None
  y_lable = None
  for x,y in validation_generator:
      if batch == (3974/batch_size):
          break
      print "predict on batch validate:",batch
      batch+=1
      if x_generator==None:
         x_generator = base_model.predict_on_batch(x)
         y_lable = y
      else:
         x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
         y_lable = np.append(y_lable,y,axis=0)
  x_generator,y_lable = shuffle(x_generator,y_lable)
  np.save(open('video_x_validate_VGG16.npy', 'w'),x_generator)
  np.save(open('video_y_validate_VGG16.npy','w'),y_lable)
  '''

  train_data = np.load(open('video_x_VGG16.npy'))
  train_labels = np.load(open('video_y_VGG16.npy'))
  train_data,train_labels = shuffle(train_data,train_labels)
  validation_data = np.load(open('video_x_validate_VGG16.npy'))
  validation_labels = np.load(open('video_y_validate_VGG16.npy'))
  validation_data,validation_labels = shuffle(validation_data,validation_labels)

  train_data = train_data.reshape(train_data.shape[0],
                     train_data.shape[1] * train_data.shape[2],
                     train_data.shape[3])
  validation_data = validation_data.reshape(validation_data.shape[0],
                     validation_data.shape[1] * validation_data.shape[2],
                     validation_data.shape[3])
  
  return train_data,train_labels,validation_data,validation_labels

def train_model(train_data,train_labels,validation_data,validation_labels):
  ''' used fully connected layers, SGD optimizer and 
      checkpoint to store the best weights'''

  model = Sequential()
  model.add(LSTM(256,dropout=0.2,input_shape=(train_data.shape[1],
                     train_data.shape[2])))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(5, activation='softmax'))
  sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  #model.load_weights('video_1_LSTM_1_512.h5')
  callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
  nb_epoch = 500
  model.fit(train_data,train_labels,validation_data=(validation_data,validation_labels),batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)
  return model

def test_on_whole_videos(train_data,train_labels,validation_data,validation_labels):
  parent = os.listdir("/Users/.../video/test")
  #.....................................Testing on whole videos.................................................................
  x = []
  y = []
  count = 0
  output = 0
  count_video = 0
  correct_video = 0
  total_video = 0
  base_model = load_VGG16_model()
  model = train_model(train_data,train_labels,validation_data,validation_labels)
  for video_class in parent[1:]:
      print video_class
      child = os.listdir("/Users/.../video/test" + "/" + video_class)
      for class_i in child[1:]:
          sub_child = os.listdir("/Users/.../video/test" + "/" + video_class + "/" + class_i)
          for image_fol in sub_child[1:]:
              if (video_class ==  'class_4' ):
                  if(count%4 == 0):
                      image = imread("/Users/.../video/test" + "/" + video_class + "/" + class_i + "/" + image_fol)
                      image = imresize(image , (224,224))

                      x.append(image)
                      y.append(output)
                      #cv2.imwrite('/Users/.../video/validate/' + video_class + '/' + str(count) + '_' + image_fol,image)
                  count+=1

              else:
                  if(count%4 == 0):
                      image = imread("/Users/.../video/test" + "/" + video_class + "/" + class_i + "/" + image_fol)
                      image = imresize(image , (224,224))
                      x.append(image)
                      y.append(output)
                      #cv2.imwrite('/Users/.../video/validate/' + video_class + '/' + str(count) + '_' + image_fol,image)
                  count+=1
          #correct_video+=1
          x = np.array(x)
          y = np.array(y)
          x_features = base_model.predict(x)
          #np.save(open('feat_' + 'class_' + str(output) + '_' + str(count_video) +'_'  + '.npy','w'),x)

          correct = 0
          
          answer = model.predict(x_features)
          for i in range(len(answer)):
              if(y[i] == np.argmax(answer[i])):
                  correct+=1
          print correct,"correct",len(answer)
          total_video+=1
          if(correct>= len(answer)/2):
              correct_video+=1
          x = []
          y = []
          count_video+=1
      output+=1

  print "correct_video",correct_video,"total_video",total_video
  print "The accuracy for video classification of ",total_video, " videos is ", (correct_video/total_video)
  
if __name__ == '__main__':
  train_generator,validation_generator = bring_data_from_directory()
  base_model = load_VGG16_model()
  train_data,train_labels,validation_data,validation_labels = extract_features_and_store(train_generator,validation_generator,base_model)
  train_model(train_data,train_labels,validation_data,validation_labels)
  test_on_whole_videos(train_data,train_labels,validation_data,validation_labels)
