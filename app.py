import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from Text_Gen import generate_text

model=tf.keras.models.load_model('text_generation_shakespeare_rnn.h5')

#print(generate_text(model, start_string="ROMEO: ", temperature=0.01))

app = Flask(__name__)

@app.route('/')
def homePage():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def main_page():
    temp=float(request.form.get('temprature'))
    textLen=int(request.form.get('TextLen'))
    inpString=request.form['inpstr']
    genText=generate_text(model, start_string=inpString,num_generate=textLen, temperature=temp)

    return render_template('index.html', genTex=genText)


if __name__=='__main__':
    app.run()