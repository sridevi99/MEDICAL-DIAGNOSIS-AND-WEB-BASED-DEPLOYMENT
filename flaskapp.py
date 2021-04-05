from __future__ import division, print_function
from flask import Flask, render_template

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from forms import ContactForm
from flask_mail import Message, Mail

mail = Mail()

app = Flask(__name__,template_folder='templates')

app.secret_key = 'development key 123456789'

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config["MAIL_USERNAME"] = 'experimentalacc64@gmail.com'
app.config["MAIL_PASSWORD"] = '#1998@EXPp'

mail.init_app(app)

model = load_model('E:/Combination/models/model_pneum.h5')

model1= load_model('E:/Combination/models/model.h5')  

model3= pickle.load(open('E:/Combination/models/model.pkl', 'rb')) 

 

lesion_classes_dict = {
     0:'Melanocytic nevi',
     1:'Melanoma',
     2:'Benign keratosis-like lesions ',
     3:'Basal cell carcinoma',
     4:'Actinic keratoses',
     5:'Vascular lesions',
     6:'Dermatofibroma'
}

model2= load_model('E:/Combination/models/model-015.model')  


label_dict={
    0:'Covid19 Negative', 
    1:'Covid19 Positive'
    }


#covid region
def model_predict2(img_path, model2):
    
    img = image.load_img(img_path, target_size=(100, 100))
    img = image.img_to_array(img)
    if(img.ndim==3):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray=img#target_size must agree with what the trained model expects!!

    # Preprocessing the image
    
    gray=gray/255
    resized=cv2.resize(gray,(100,100))
    reshaped=resized.reshape(1,100,100)

   
    cpreds = model2.predict(reshaped)
    return cpreds



#pneumonia region
def model_predict(img_path, model):
    
    img = image.load_img(img_path, target_size=(200, 200),color_mode='grayscale') #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    resized_arr=cv2.resize(img,(200,200))
    resized_arr=resized_arr.reshape(-1,200,200,1)

   
    preds = model.predict(resized_arr)
    return preds

#skin region
def model_predict1(img_path, model1):
    img = image.load_img(img_path, target_size=(128,128,3))
  
    #img = np.asarray(pil_image.open('img').resize((120,120)))
    #x = np.asarray(img.tolist())

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    spreds = model1.predict(x.reshape(1,128,128,3))
    return spreds
    

@app.route('/')
def home():
   return render_template('newhome.html')
@app.route('/aboutus')
def aboutus():
    title = 'About Us'
    return (render_template('aboutus.html',title=title))

@app.route('/projectabout')
def projectabout():
   return render_template('projectdoc.html')


@app.route('/homeflask')
def homeflask():
   return render_template('homeflask.html')

@app.route('/contactflask', methods=['GET', 'POST'])
def contactflask():
  form = ContactForm()

  if request.method == 'POST':
    if form.validate() == False:
      flash('All fields are required.')
      return render_template('contactflask.html', form=form)
    else:
      msg = Message(form.subject.data, sender='contact@example.com', recipients=['experimentalacc64@gmail.com'])
      msg.body = """
      From: %s <%s>
      %s
      """ % (form.name.data, form.email.data, form.message.data)
      mail.send(msg)

      return render_template('contactflask.html', success=True)

  elif request.method == 'GET':
    return render_template('contactflask.html', form=form)


@app.route('/covid_page', methods = ['GET','POST'])
def covid_page():
    title='Covid Detection'
    
    return(render_template("cindex.html", title=title))

@app.route('/covid_predict', methods=['POST'])
def upload2():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        cpreds = model_predict2(file_path , model2)
        os.remove(file_path)
       
        result=np.argmax(cpreds,axis=1)[0]
        result=label_dict[result]

        return result
    
@app.route('/heart_ml')
def heart_ml():
    return render_template('Heart Disease Classifier.html')

@app.route('/predict_hdc', methods =['POST'])
def predict_hdc():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model3.predict(array_features)
    
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('Heart Disease Classifier.html', 
                               result = 'You may not likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html', 
                               result = 'You may likely to have heart disease!')


@app.route('/skin_page', methods = ['GET','POST'])
def skin_page():
    title = 'Skin page'
    return render_template('index1.html', title=title)

@app.route('/predict-skin', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        spreds = model_predict1(file_path , model1)
        os.remove(file_path)
        
        top3 = np.argsort(spreds[0])[:-4:-1]
        result =[]
        for i in range(2):
           a = ("{}".format(lesion_classes_dict[top3[i]]))
       
        result.append(a) 
        result = str(result)
        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        #pr = lesion_classes_dict[pred_class[0]]
        #result =str(pr)         
        return result
    #return None



@app.route('/pnum_page', methods = ['GET','POST'])
def pnum_page():
    title = 'Second page'
    return render_template('pindex.html', title=title)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str1 = ' Patient Normal'
        str2 = ' Patient has Pneumonia'
        if (preds[0] == 1.):
            return str1
        else:
            return str2
    return None

if __name__ == '__main__':
        app.run(debug=True)