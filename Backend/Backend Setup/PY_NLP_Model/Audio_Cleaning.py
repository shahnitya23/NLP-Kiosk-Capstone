#!/usr/bin/env python
# coding: utf-8

# In[16]:


# import libraries
import speech_recognition as sr
import numpy as np

from langdetect import detect_langs

from collections import defaultdict
from collections import Counter

# %reset -f


# In[17]:


# import libraries
# import speech_recognition as sr
# import numpy as np

# from langdetect import detect_langs

# from collections import defaultdict
# from collections import Counter


# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()
with sr.Microphone() as source:
    audio_text = r.listen(source)

    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:

        # Using google speech recognition
        text_en = r.recognize_google(audio_text)
        # print('Converting audio transcripts into text ...')
        # print(text_en)
        get_ipython().run_line_magic("store", "text_en")
        # print()

        # Adding French langauge option
        text_fr = r.recognize_google(audio_text, language="fr-FR")
        # print('Converting audio transcripts into French text ...')
        # print(text_fr)
        get_ipython().run_line_magic("store", "text_fr")
        # print()

        # Adding Spanish langauge option
        text_es = r.recognize_google(audio_text, language="es-AR")
        # print('Converting audio transcripts into Spanish text ...')
        # print(text_es)
        get_ipython().run_line_magic("store", "text_es")
        # print()

        # # Adding Hindi langauge option
        # text_hi = r.recognize_google(audio_text, language = "hi-IN")
        # print('Converting audio transcripts into Spanish text ...')
        # print(text_hi)
        # print()

    except:
        print("Sorry.. run again...")


# In[18]:


from langdetect import detect_langs


def lang_detect(response1, response2, response3):
    language_prob = []

    # Make into String Elements for Data Cleaning
    detect_en = [str(i) for i in detect_langs(response1)]
    language_prob.append(detect_en)
    # print('English: ', language_prob)

    detect_fr = [str(i) for i in detect_langs(response2)]
    language_prob.append(detect_fr)
    # print('French: ', language_prob)

    detect_es = [str(i) for i in detect_langs(response3)]
    language_prob.append(detect_es)
    # print('Spanish: ', language_prob)

    # print(language_prob)
    return language_prob


# In[19]:


def lang_prob(lang_probability):
    # language_prob = []
    language_code = []
    lang_code = []

    # language_prob = [['es:0.9999932356592219'], ['es:0.9999941641326395'], ['es:0.999996766257174'], ['ne:0.5714284946155643', 'hi:0.4285697429223616']]

    # Seperate Probs and Language Codes into 2 lists
    for i in range(0, len(lang_probability)):

        for j in range(0, len(lang_probability[i])):

            # Language Code List of Lists
            code = lang_probability[i][j][0:2]
            lang_code.append(code)

            # Language Probability
            lang_probability[i][j] = lang_probability[i][j][3:]

        language_code.append(lang_code)

    # Make into One List instead of Lists of Lists
    flat_list = []
    for sublist in lang_probability:
        for item in sublist:
            flat_list.append(item)

    # Convert the probabilities from String to Float
    probs = []
    for item in flat_list:
        probs.append(float(item))

    # print('Language Code: ', lang_code)
    # print('Language Probability: ', probs)
    return lang_code, probs


# In[20]:


def ISO_639(langauge_code, probability):
    lang_counter = Counter(langauge_code)

    # extracting value to compare
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
        # print('Original Dictionary: ', dict(d))

        # Value list mean
        mean_probs = {key: np.mean(val) for key, val in dict(d).items()}
        # print('Transformed Dictionary: ', mean_probs)

        ISO_639_2 = str(max(mean_probs, key=mean_probs.get))
        # print('Langauge ISO 639-2: ', ISO_639_2)

    else:
        ISO_639_2 = max(lang_counter, key=lang_counter.get)
        # print('Lang Counter: ', Counter(langauge_code))
        # print('Langauge ISO 639-2: ', ISO_639_2)

    return ISO_639_2
