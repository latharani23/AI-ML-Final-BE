from flask import Flask, render_template, url_for, request
import sqlite3
import shutil
import os
import sys
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt

from twilio.rest import Client

account_sid="AC96ee88c73a7f5e022ba6400a2a009f74"
auth_token="adbc485e319a34834a91ed6eb6e67958"
client=Client(account_sid,auth_token)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/lung')
def lung():
    return render_template('lung.html')

@app.route('/lung_disease', methods=['GET', 'POST'])
def leaf_disease():
    if request.method == 'POST':
        image = request.form['img']

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Lungdisease-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
            verifying_data = []
            path = 'static/test/'+image
            img_num = image.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 6, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        # fig = plt.figure()
        
        accuracy=""
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            # y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = " Lung Cancer"
                print("The predicted image of the Cancer is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the Cancer is with a accuracy of {}%".format(model_out[0]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])
                F=float(model_out[5])
                dic={'Lung Cancer':A,'Viral Pneumonia':B,'Covid-19':C,'Tuberculosis':D,'Normal':E,'unwanted':F}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Lung disease detection....")
                plt.savefig('static/matrix.png')
                client.api.account.messages.create(
					                    to="+91-8088203082",
                                        from_="+15077103862",
                                        body="Hello, you have been diagnosed with Lung Cancer. For proper treatment and consultation, visit 1. Fortis Cancer Institute - Bannerghatta, Bengaluru | 2.HCG Comprehensive Cancer Care Hospital - K.R. Road, Bengaluru.")
                
                
            elif np.argmax(model_out) == 1:
                str_label  = "Viral Pneumonia"
                print("The predicted image of the Viral Pneumonia is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the Viral Pneumonia is with a accuracy of {}%".format(model_out[1]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])
                F=float(model_out[5])
                dic={'Lung Cancer':A,'Viral Pneumonia':B,'Covid-19':C,'Tuberculosis':D,'Normal':E,'unwanted':F}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Lung disease detection....")
                plt.savefig('static/matrix.png')
                client.api.account.messages.create(
										to="+91-8088203082",
                                        from_="+15077103862",
                                        body="Hello, you have been diagnosed with Viral Pneumonia. For proper treatment and consultation, visit 1. Manipal Hospital - HAL Road, Bengaluru | 2. BGS Gleneagles Global Hospital - Kengeri, Bengaluru.")
                
                
            elif np.argmax(model_out) == 2:
                str_label = "Covid-19"
                print("The predicted image of the Covid is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the Covid is with a accuracy of {}%".format(model_out[2]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])
                F=float(model_out[5])
                dic={'Lung Cancer':A,'Viral Pneumonia':B,'Covid-19':C,'Tuberculosis':D,'Normal':E,'unwanted':F}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Lung disease detection....")
                plt.savefig('static/matrix.png')
                client.api.account.messages.create(
					                    to="+91-8088203082",
                                        from_="+15077103862",
                                        body="Hello, you have been diagnosed with Covid-19. For proper treatment and consultation, visit 1. Bowring Medical Hospital - Shivajinagar, Bengaluru | 2. Victoria Hospital - New Taragupet, Bengaluru.")
                            
            elif np.argmax(model_out) == 3:
                str_label = "Tuberculosis"
                print("The predicted image of the Tuberculosis is with a accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the Tuberculosis is with a accuracy of {}%".format(model_out[3]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])
                F=float(model_out[5])
                dic={'Lung Cancer':A,'Viral Pneumonia':B,'Covid-19':C,'Tuberculosis':D,'Normal':E,'unwanted':F}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Lung disease detection....")
                plt.savefig('static/matrix.png')
                client.api.account.messages.create(
					                    to="+91-8088203082",
                                        from_="+15077103862",
                                        body="Hello, you have been diagnosed with Tuberculosis. For proper treatment and consultation, visit 1. St. Johns Medical College and Hospital - Koramangla, Bengaluru | 2. SDS Tuberculosis Research Center & Rajiv Gandhi Institute of Chest Diseases - Hosur Road, Bengaluru.")


            elif np.argmax(model_out) == 4:
                str_label = "Normal"
                print("The predicted image of the Normal is with a accuracy of {} %".format(model_out[4]*100))
                accuracy="The predicted image of the Normal is with a accuracy of {}%".format(model_out[4]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])
                F=float(model_out[5])
                dic={'Lung Cancer':A,'Viral Pneumonia':B,'Covid-19':C,'Tuberculosis':D,'Normal':E,'unwanted':F}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Lung disease detection....")
                plt.savefig('static/matrix.png')
                client.api.account.messages.create(
					                    to="+91-8088203082",
                                        from_="+15077103862",
                                        body="Hello, you have not been tested positive and your results turn out to be Normal.")
            elif np.argmax(model_out) == 5:
                str_label = "Inappropriate image. Add an X-Ray image"
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                E=float(model_out[4])
                F=float(model_out[5])
                dic={'Lung Cancer':A,'Viral Pneumonia':B,'Covid-19':C,'Tuberculosis':D,'Normal':E,'unwanted':F}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Lung disease detection....")
                plt.savefig('static/matrix.png')
                


        return render_template('lung.html', status=str_label,accuracy=accuracy,ImageDisplay="http://127.0.0.1:5000/static/test/"+image,ImageDisplay1="http://127.0.0.1:5000/static/matrix.png")

    return render_template('lung.html')


@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
