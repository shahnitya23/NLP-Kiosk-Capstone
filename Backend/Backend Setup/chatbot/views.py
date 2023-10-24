# These libraries are used by Django for rendering your pages.
from django.http import HttpResponse
from django.shortcuts import render, redirect

import numpy as np
import random
import os
import json
import pickle
import time

import speech_recognition as sr
from langdetect import detect_langs

from collections import defaultdict, Counter

import nltk
nltk.download('all')
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from deep_translator import GoogleTranslator

import pyttsx3


def lemmatizer_initialize():
    lemmatizer = WordNetLemmatizer()
    return lemmatizer


def load_words_pkl():
    file_path = os.path.join(os.path.dirname(__file__), 'words.pkl')

    with open(file_path, 'rb') as file:
        words = pickle.load(file)

    return words


def load_classes_pkl():
    file_path = os.path.join(os.path.dirname(__file__), 'classes.pkl')

    with open(file_path, 'rb') as file_classes:
        classes = pickle.load(file_classes)

    return classes


def load_intents():
    file_path = os.path.join(os.path.dirname(__file__), 'Intent_English.json')
    with open(file_path, 'r') as file:
        intents = json.load(file)
    return intents

def training(request):
    intents = load_intents()

    lemmatizer = WordNetLemmatizer()

    # Empty Lists
    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ".", ",", "(", ")"]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:

            word_list = nltk.word_tokenize(pattern) 
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))

            if (intent["tag"] not in classes):
                classes.append(intent["tag"]) 

    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]  
    words = sorted(set(words))  
    classes = sorted(set(classes))

    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.asarray(training, dtype="object")

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = Sequential()

    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))
    
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=sgd, metrics=["accuracy"])
    
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    model.save("chatbotmodel.h5", hist)

    return HttpResponse('Created words.pkl.<br>Created classes.pkl.<br>Saved NLP Model (chatbotmodel.h5).<br>Training of the NLP Model Completed!')


def cleaning_up_sentence(sentence):
    lemmatizer = lemmatizer_initialize()

    sentence_words = nltk.word_tokenize(sentence)                               
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]    
    return sentence_words                                                       


def bag_of_words(sentence):                             
    words = load_words_pkl()
    sentence_words = cleaning_up_sentence(sentence)     
    bag = [0] * len(words)                              

    for w in sentence_words:                            
        for i, word in enumerate(words):                
            if word == w:                               
                bag[i] = 1                              
    
    return np.array(bag)                                


def predict_class(sentence):
    classes = load_classes_pkl()
    file_path = os.path.join(os.path.dirname(__file__), 'chatbotmodel.h5')
    model = load_model(file_path)

    bow = bag_of_words(sentence)                                                
    res = model.predict(np.array([bow]))[0]                                     

    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]         

    result.sort(key = lambda x: x[1], reverse = True)                           
    return_list = []                                                            

    for r in result:                                                            
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])}) 

    return return_list                                                          


def get_response(intents_list, intents_json):       
    tag = intents_list[0]['intent']                 
    list_of_intents = intents_json['intents']       

    for i in list_of_intents:                       
        if i['tag'] == tag:                         
            result = random.choice(i['responses'])  
            break                                   
    
    return result                                   


def capture_and_recognize(request):
    if request.method == 'GET':
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio_text = r.listen(source)
            try:
                # Using google speech recognition
                text_en = r.recognize_google(audio_text)
                # Adding French language option
                text_fr = r.recognize_google(audio_text, language="fr-FR")
                # Adding Spanish language option
                text_es = r.recognize_google(audio_text, language="es-AR")

                return (text_en, text_fr, text_es)

            except sr.UnknownValueError:
                return HttpResponse("Could not understand audio. Please press the button again to try again!")
            
            except sr.RequestError as e:
                return HttpResponse(f"Could not request results from Google Speech Recognition service; {e}")
            
    return HttpResponse("Method Not Allowed")


def lang_detect(response1, response2, response3):
    language_prob = []

    # Make into String Elements for Data Cleaning
    detect_en = [str(i) for i in detect_langs(response1)]
    language_prob.extend(detect_en)

    detect_fr = [str(i) for i in detect_langs(response2)]
    language_prob.extend(detect_fr)

    detect_es = [str(i) for i in detect_langs(response3)]
    language_prob.extend(detect_es)

    return list(language_prob)


def lang_prob(lang_probability):
    language_code = []

    for i in range(0, len(lang_probability)):         
        code = lang_probability[i][0:2]
        language_code.append(code)

        lang_probability[i] = lang_probability[i][3:]
    
    probs = [float(i) for i in lang_probability]

    return language_code, probs


def ISO_639(langauge_code, probability):
    lang_counter = Counter(langauge_code)

    test_val = list(lang_counter.values())[0]
    res = True
    
    for i in lang_counter:
        if lang_counter[i] != test_val:
            res = False
            break

    if res == True:
        d = defaultdict(list)
        for key, value in zip(langauge_code, probability):
            d[key].append(value)

        mean_probs = {key: np.mean(val) for key, val in dict(d).items()}

        ISO_639_2 = str(max(mean_probs, key = mean_probs.get))

    else:
        ISO_639_2 = max(lang_counter, key = lang_counter.get)
    
    return ISO_639_2


# When the Button is Pressed
def main(request):

    intents = load_intents()

    English, French, Spanish = capture_and_recognize(request)
    possible_langs = lang_detect(English, French, Spanish)
    lang_list, prob_list = lang_prob(possible_langs)
    lang_ISO = ISO_639(lang_list, prob_list)

    if lang_ISO == 'en':
        message = English
        
        if message == 'exit please':
            details = {"User_Request": 'User has exited BableBot. Please press the button to speak to BabelBot!'}
            quit

        time.sleep(1)

        ints = predict_class(message)
        res = get_response(ints, intents)
        details = {
            "User_Request": English,
            "Chatbot_Response": res
            }
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.say(res)
        engine.runAndWait()

    elif lang_ISO == 'fr':
        message = French

        time.sleep(1)

        translator_fr2en = GoogleTranslator(source = 'auto', target = 'en')
        fr2en = translator_fr2en.translate(message)

        if fr2en.lower() == 'exit please':
            details = {"User_Request": 'User has exited BableBot. Please press the button to speak to BabelBot!'}
            quit
        
        ints = predict_class(fr2en)
        res = get_response(ints, intents)
        translator_en2fr = GoogleTranslator(source = 'auto', target = 'fr')
        res_en2fr = translator_en2fr.translate(res)
        details = {
            "User_Request": French,
            "Chatbot_Response": res_en2fr
            }
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[3].id)
        engine.say(res_en2fr)
        engine.runAndWait()


    elif lang_ISO == 'es':
        message = Spanish

        time.sleep(1)

        translator_es2en = GoogleTranslator(source = 'auto', target = 'en')
        es2en = translator_es2en.translate(message)

        if es2en.lower() == 'exit please':
            details = {"User_Request": 'User has exited BableBot. Please press the button to speak to BabelBot!'}
            quit
        
        ints = predict_class(es2en)
        res = get_response(ints, intents)
        translator_en2es = GoogleTranslator(source = 'auto', target = 'es')
        res_en2es = translator_en2es.translate(res)
        details = {
            "User_Request": Spanish,
            "Chatbot_Response": res_en2es
            }
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[2].id)
        engine.say(res_en2es)
        engine.runAndWait()
    
    else:
        return HttpResponse("Could not understand audio. Please press the button again to try again!")

    return render(request,"kiosk.html",details)