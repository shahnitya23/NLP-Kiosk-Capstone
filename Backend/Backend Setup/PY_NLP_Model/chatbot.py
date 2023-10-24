#!/usr/bin/env python
# coding: utf-8

# ## **ChatBot**

# #### **Importing Libraries**

# In[36]:


import random
import json
import pickle
import numpy as np

import pyttsx3

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import nbimporter
import Audio_Cleaning

from deep_translator import GoogleTranslator

import import_ipynb
import time


# In[37]:


lemmatizer = (
    WordNetLemmatizer()
)  # creating instance of lemmatizer - reducing a word to its base or dictionary form

intents = json.loads(open("Intent English.json").read())

# From training.ipynb
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")


# #### **Function for Cleaning Up the Sentences**

# In[38]:


def cleaning_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # splits a given sentence into words
    sentence_words = [
        lemmatizer.lemmatize(word) for word in sentence_words
    ]  # for each word reduce to its base form
    return sentence_words  # return list of words reduced to its base form


# #### **Function for Bag-of-Words**

# In[39]:


def bag_of_words(
    sentence,
):  # A representation of text that describes the occurrence of words within the sentence
    sentence_words = cleaning_up_sentence(
        sentence
    )  # Go to the function above to get a list of words reduced to its base form
    bag = [0] * len(words)  # List of 0 length of words (from training.ipynb)

    for w in sentence_words:  # for word in list of words reduced to its base form
        for i, word in enumerate(words):  # i = iteration, w = list element (word)
            if word == w:  # if the words in both lists match
                bag[i] = 1  # change particular index in bag from 0 to 1

    return np.array(bag)  # return array of bag


# #### **Function for Predicting**

# In[40]:


def predict_class(sentence):
    bow = bag_of_words(
        sentence
    )  # Go to the function above to get a representation of text that describes the occurrence of words within the sentence
    res = model.predict(np.array([bow]))[
        0
    ]  # Based of the array of bag-of-words get the model to predict the response

    ERROR_THRESHOLD = 0.25
    result = [
        [i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD
    ]  # If r is greater than the error threshold, get the index and the response probability and make into a list item

    result.sort(
        key=lambda x: x[1], reverse=True
    )  # Sort the list from highest probability first
    return_list = []  # Empty list

    for r in result:  # For each potential result in result list
        return_list.append(
            {"intent": classes[r[0]], "probability": str(r[1])}
        )  # Get the tag (from .json file) and the probability of the response and append as list item to the empty list created above

    return return_list  # Return the list with tag and probability


# #### **Getting Response Function**

# In[41]:


def get_response(
    intents_list, intents_json
):  # Get the list with tag and probability & the .json file
    tag = intents_list[0][
        "intent"
    ]  # Get the tag of the first element of the list with tag and prob.
    list_of_intents = intents_json[
        "intents"
    ]  # Get the contents of the .jsn that fall under intents (basically all the info comprised within)

    for i in list_of_intents:  # Go section by section
        if (
            i["tag"] == tag
        ):  # If tag matches the one found above in the first element of the list with tag and prob
            result = random.choice(
                i["responses"]
            )  # Randomly choose a response from that tag section
            break  # Break and exist the loop

    return result  # Return the randomly choosen response


# #### **Main**

# In[ ]:


languages = ["en", "fr", "es"]  # List of potential languages
print("GO! Bot is running!")

# while True:
if __name__ == "__main__":
    # %run ./Testing.ipynb
    # %run ./Speech_Lang_Detector.ipynb
    print("Speak Now...")
    get_ipython().run_line_magic("run", "./Audio_Cleaning.ipynb")

    possible_langs = Audio_Cleaning.lang_detect(text_en, text_fr, text_es)
    lang_list, prob_list = Audio_Cleaning.lang_prob(possible_langs)
    lang_ISO = Audio_Cleaning.ISO_639(lang_list, prob_list)

    # English
    if languages.index(lang_ISO) == 0:
        message = text_en
        print("You: {}".format(message))

        if message.lower() == "exit please":
            break

        time.sleep(1)

        ints = predict_class(message)
        res = get_response(ints, intents)
        print("Bot: {}\n".format(res))

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[1].id)
        engine.say(res)
        engine.runAndWait()

    # French
    elif languages.index(lang_ISO) == 1:
        message = text_fr
        print("You: {}".format(message))

        time.sleep(1)

        translator_fr2en = GoogleTranslator(source="auto", target="en")
        fr2en = translator_fr2en.translate(message)
        print("Translated FR 2 EN: ", translator_fr2en.translate(message))

        if fr2en.lower() == "exit please":
            break

        ints = predict_class(fr2en)
        res = get_response(ints, intents)
        translator_en2fr = GoogleTranslator(source="auto", target="fr")
        res_en2fr = translator_en2fr.translate(res)
        print("Bot: {}\n".format(res_en2fr))

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[3].id)
        engine.say(res_en2fr)
        engine.runAndWait()

    # Spanish
    # if languages.index(lang_ISO) == 2:
    else:
        message = text_es
        print("You: {}".format(message))

        time.sleep(1)

        translator_es2en = GoogleTranslator(source="auto", target="en")
        es2en = translator_es2en.translate(message)
        print("Translated ES 2 EN: ", translator_es2en.translate(message))

        if es2en.lower() == "exit please":
            break

        ints = predict_class(es2en)
        res = get_response(ints, intents)
        translator_en2es = GoogleTranslator(source="auto", target="es")
        res_en2es = translator_en2es.translate(res)
        print("Bot: {}\n".format(res_en2es))

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[2].id)
        engine.say(res_en2es)
        engine.runAndWait()

    time.sleep(2)


# In[ ]:
