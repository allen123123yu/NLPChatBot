# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:20:43 2023

@author: user
"""

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import time

with open(r"C:\Users\user\Downloads\First Aid Recommendations Intents\intentsca.json") as f:
  data_ca = json.load(f)

words_ca = []
labels_ca = []
docs_x_ca = []
docs_y_ca = []

for intent_ca in data_ca["intents"]:
        for pattern_ca in intent_ca["patterns"]:
            wrds_ca = nltk.word_tokenize(pattern_ca)
            words_ca.extend(wrds_ca)
            docs_x_ca.append(wrds_ca)
            docs_y_ca.append(intent_ca["tag"])

        if intent_ca["tag"] not in labels_ca:
            labels_ca.append(intent_ca["tag"])

words_ca = [stemmer.stem(w_ca.lower()) for w_ca in words_ca if w_ca != "?"]
words_ca = sorted(list(set(words_ca)))

labels_ca = sorted(labels_ca)


training = []
output = []

out_empty = [0 for _ in range(len(labels_ca))]

for x_ca, doc in enumerate(docs_x_ca):
        bag_ca = []

        wrds_ca = [stemmer.stem(w_ca) for w_ca in doc]

        for w_ca in words_ca:
            if w_ca in wrds_ca:
                bag_ca.append(1)
            else:
                bag_ca.append(0)
        output_row = out_empty[:]

        output_row[labels_ca.index(docs_y_ca[x_ca])] = 1

        training.append(bag_ca)
        output.append(output_row)

training_np_arr = numpy.array(training)
output_np_arr= numpy.array(output)

tensorflow.compat.v1.reset_default_graph()

net_tflearn_input_data_shape_list=[None, len(training_np_arr[0])]
net_tflearn_fully_connected_1=8
net_tflearn_fully_connected_2=8
net_tflearn_fully_connected_3=len(output_np_arr[0])
net_tflearn_fully_connected_4_activation="softmax"

net = tflearn.input_data(shape=net_tflearn_input_data_shape_list)
net = tflearn.fully_connected(net, net_tflearn_fully_connected_1)
net = tflearn.fully_connected(net, net_tflearn_fully_connected_2)
net = tflearn.fully_connected(net, net_tflearn_fully_connected_3, activation=net_tflearn_fully_connected_4_activation)
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
        model.load("model_ca.tflearn")
except:
        model = tflearn.DNN(net)
        model.fit(training_np_arr, output_np_arr, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model_ca.tflearn")

def bag_of_words_ca(s_ca,words_ca):
        bag_ca = [0 for _ in range(len(words_ca))]

        s_words_ca = nltk.word_tokenize(s_ca)
        s_words_stemmed_ca=[stemmer.stem(word_ca.lower()) for word_ca in s_words_ca]

        for se_ca in s_words_stemmed_ca:
            for i, w_ca in enumerate(words_ca):
                if w_ca == se_ca:
                    bag_ca[i] = 1

        return numpy.array(bag_ca)


            
def chatca():
    count = 0
    print("911 what is the nature of your emergency? We offer fire, police and first aid services. Type I’m fine, and we’ll end this and check back in on you. (type quit to exit)")
    while True:
        inp_ca = input("You: ")
        if inp_ca.lower() in ["quit"]:
            print("Have a nice day.")
            break      
        if inp_ca.lower() in ["i'm fine", "im fine", "imfine"]:
            print("Have a nice day.")
            time.sleep(5)
            chatcheckca()
            break
        else:
            results_ca = model.predict([bag_of_words_ca(inp_ca,words_ca)])[0]
            results_index_ca = numpy.argmax(results_ca)
            tag_ca = labels_ca[results_index_ca]
                
            if results_ca[results_index_ca] > 0.5:
                for tg_ca in data_ca["intents"]:
                    if tg_ca['tag'] == tag_ca:
                        responses = tg_ca['responses']
                print(random.choice(responses))
                print("\n")
            else:
                if count == 3: 
                    print("I couldn't understand you, please sit tight, help is on the way.")
                    break
                print("Sorry I couldn't understand that, please try again.")
                count += 1
def chatcheckca():
    count = 0
    print("911 calling back to check in on you. If you're fine, please reply all is well. Type I'm fine and we'll send help immediately. (type quit to exit)")
    while True:
        inp_ch = input("You: ")
        if inp_ch.lower() in ["all is well", "alls well", "allswell", "all'swell"]:
            print("Have a nice day.")
            break
        if inp_ch.lower() in ["i'm fine", "im fine", "imfine"]:
            print("Have a nice day.")
        else:
            if count == 3: 
                print("We're not sure if you're pranking us. Regardless, police are on their way.")
                break
            print("Sorry I couldn't understand that, please try again.")
            count += 1   
                    
    return
chatca()
