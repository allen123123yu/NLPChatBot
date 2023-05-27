import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow.compat.v1 as tf
import random
import json
import time

nltk.download('punkt')

stemmer = LancasterStemmer()

with open(r"C:\Users\user\Downloads\First Aid Recommendations Intents\faintents.json") as f:
    data_fa = json.load(f)

words_fa = []
labels_fa = []
docs_x_fa = []
docs_y_fa = []

for intent_fa in data_fa["intents"]:
    for pattern_fa in intent_fa["patterns"]:
        wrds_fa = nltk.word_tokenize(pattern_fa)
        words_fa.extend(wrds_fa)
        docs_x_fa.append(wrds_fa)
        docs_y_fa.append(intent_fa["tag"])

    if intent_fa["tag"] not in labels_fa:
        labels_fa.append(intent_fa["tag"])

words_fa = [stemmer.stem(w_fa.lower()) for w_fa in words_fa if w_fa != "?"]
words_fa = sorted(list(set(words_fa)))

labels_fa = sorted(labels_fa)

training_fa = []
output_fa = []

out_empty_fa = [0 for _ in range(len(labels_fa))]

for x_fa, doc in enumerate(docs_x_fa):
    bag_fa = []
    wrds_fa = [stemmer.stem(w_fa) for w_fa in doc]

    for w_fa in words_fa:
        if w_fa in wrds_fa:
            bag_fa.append(1)
        else:
            bag_fa.append(0)
    output_row_fa = out_empty_fa[:]

    output_row_fa[labels_fa.index(docs_y_fa[x_fa])] = 1

    training_fa.append(bag_fa)
    output_fa.append(output_row_fa)

training_np_arr_fa = np.array(training_fa)
output_np_arr_fa = np.array(output_fa)

tf.reset_default_graph()

net_tflearn_input_data_shape_list = [None, len(training_np_arr_fa[0])]
net_tflearn_fully_connected_1 = 8
net_tflearn_fully_connected_2 = 8
net_tflearn_fully_connected_3 = len(output_np_arr_fa[0])
net_tflearn_fully_connected_4_activation = "softmax"

net_fa = tflearn.input_data(shape=net_tflearn_input_data_shape_list)
net_fa = tflearn.fully_connected(net_fa, net_tflearn_fully_connected_1)
net_fa = tflearn.fully_connected(net_fa, net_tflearn_fully_connected_2)
net_fa = tflearn.fully_connected(net_fa, net_tflearn_fully_connected_3, activation=net_tflearn_fully_connected_4_activation)
net_fa = tflearn.regression(net_fa)

model_fa = tflearn.DNN(net_fa)

try:
    model_fa.load("model_fa.tflearn")
except:
    model_fa = tflearn.DNN(net_fa)
    model_fa.fit(training_np_arr_fa, output_np_arr_fa, n_epoch=1000, batch_size=8, show_metric=True)
    model_fa.save("model_fa.tflearn")

def bag_of_words_fa(s_fa, words_fa):
    bag_fa = [0 for _ in range(len(words_fa))]

    s_words_fa = nltk.word_tokenize(s_fa)
    s_words_stemmed_fa = [stemmer.stem(word_fa.lower()) for word_fa in s_words_fa]

    for se_fa in s_words_stemmed_fa:
        for i, w_fa in enumerate(words_fa):
            if w_fa == se_fa:
                bag_fa[i] = 1

    return np.array(bag_fa)

with open(r"C:\Users\user\Downloads\First Aid Recommendations Intents\pointents.json") as f:
    data_po = json.load(f)

words_po = []
labels_po = []
docs_x_po = []
docs_y_po = []

for intent_po in data_po["intents"]:
    for pattern_po in intent_po["patterns"]:
        wrds_po = nltk.word_tokenize(pattern_po)
        words_po.extend(wrds_po)
        docs_x_po.append(wrds_po)
        docs_y_po.append(intent_po["tag"])

    if intent_po["tag"] not in labels_po:
        labels_po.append(intent_po["tag"])

words_po = [stemmer.stem(w_po.lower()) for w_po in words_po if w_po != "?"]
words_po = sorted(list(set(words_po)))

labels_po = sorted(labels_po)

training_po = []
output_po = []

out_empty_po = [0 for _ in range(len(labels_po))]

for x_po, doc in enumerate(docs_x_po):
    bag_po = []
    wrds_po = [stemmer.stem(w_po) for w_po in doc]

    for w_po in words_po:
        if w_po in wrds_po:
            bag_po.append(1)
        else:
            bag_po.append(0)
    output_row_po = out_empty_po[:]

    output_row_po[labels_po.index(docs_y_po[x_po])] = 1

    training_po.append(bag_po)
    output_po.append(output_row_po)

training_np_arr_po = np.array(training_po)
output_np_arr_po = np.array(output_po)

tf.reset_default_graph()

net_tflearn_input_data_shape_list = [None, len(training_np_arr_po[0])]
net_tflearn_fully_connected_1 = 8
net_tflearn_fully_connected_2 = 8
net_tflearn_fully_connected_3 = len(output_np_arr_po[0])
net_tflearn_fully_connected_4_activation = "softmax"

net_po = tflearn.input_data(shape=net_tflearn_input_data_shape_list)
net_po = tflearn.fully_connected(net_po, net_tflearn_fully_connected_1)
net_po = tflearn.fully_connected(net_po, net_tflearn_fully_connected_2)
net_po = tflearn.fully_connected(net_po, net_tflearn_fully_connected_3, activation=net_tflearn_fully_connected_4_activation)
net_po = tflearn.regression(net_po)

model_po = tflearn.DNN(net_po)

try:
    model_po.load("model_po.tflearn")
except:
    model_po = tflearn.DNN(net_po)
    model_po.fit(training_np_arr_po, output_np_arr_po, n_epoch=1000, batch_size=8, show_metric=True)
    model_po.save("model_po.tflearn")

def bag_of_words_po(s_po, words_po):
    bag_po = [0 for _ in range(len(words_po))]

    s_words_po = nltk.word_tokenize(s_po)
    s_words_stemmed_po = [stemmer.stem(word_po.lower()) for word_po in s_words_po]

    for se_po in s_words_stemmed_po:
        for i, w_po in enumerate(words_po):
            if w_po == se_po:
                bag_po[i] = 1

    return np.array(bag_po)

with open(r"C:\Users\user\Downloads\First Aid Recommendations Intents\ffintentsmids.json") as f:
    data_ff = json.load(f)

words_ff = []
labels_ff = []
docs_x_ff = []
docs_y_ff = []

for intent_ff in data_ff["intents"]:
    for pattern_ff in intent_ff["patterns"]:
        wrds_ff = nltk.word_tokenize(pattern_ff)
        words_ff.extend(wrds_ff)
        docs_x_ff.append(wrds_ff)
        docs_y_ff.append(intent_ff["tag"])

    if intent_ff["tag"] not in labels_ff:
        labels_ff.append(intent_ff["tag"])

words_ff = [stemmer.stem(w_ff.lower()) for w_ff in words_ff if w_ff != "?"]
words_ff = sorted(list(set(words_ff)))

labels_ff = sorted(labels_ff)

training_ff = []
output_ff = []

out_empty_ff = [0 for _ in range(len(labels_ff))]

for x_ff, doc in enumerate(docs_x_ff):
    bag_ff = []
    wrds_ff = [stemmer.stem(w_ff) for w_ff in doc]

    for w_ff in words_ff:
        if w_ff in wrds_ff:
            bag_ff.append(1)
        else:
            bag_ff.append(0)
    output_row_ff = out_empty_ff[:]

    output_row_ff[labels_ff.index(docs_y_ff[x_ff])] = 1

    training_ff.append(bag_ff)
    output_ff.append(output_row_ff)

training_np_arr_ff = np.array(training_ff)
output_np_arr_ff = np.array(output_ff)

tf.reset_default_graph()

net_tflearn_input_data_shape_list = [None, len(training_np_arr_ff[0])]
net_tflearn_fully_connected_1 = 8
net_tflearn_fully_connected_2 = 8
net_tflearn_fully_connected_3 = len(output_np_arr_ff[0])
net_tflearn_fully_connected_4_activation = "softmax"

net_ff = tflearn.input_data(shape=net_tflearn_input_data_shape_list)
net_ff = tflearn.fully_connected(net_ff, net_tflearn_fully_connected_1)
net_ff = tflearn.fully_connected(net_ff, net_tflearn_fully_connected_2)
net_ff = tflearn.fully_connected(net_ff, net_tflearn_fully_connected_3, activation=net_tflearn_fully_connected_4_activation)
net_ff = tflearn.regression(net_ff)

model_ff = tflearn.DNN(net_ff)

try:
    model_ff.load("model_ff.tflearn")
except:
    model_ff = tflearn.DNN(net_ff)
    model_ff.fit(training_np_arr_ff, output_np_arr_ff, n_epoch=1000, batch_size=8, show_metric=True)
    model_ff.save("model_ff.tflearn")

def bag_of_words_ff(s_ff, words_ff):
    bag_ff = [0 for _ in range(len(words_ff))]

    s_words_ff = nltk.word_tokenize(s_ff)
    s_words_stemmed_ff = [stemmer.stem(word_ff.lower()) for word_ff in s_words_ff]

    for se_ff in s_words_stemmed_ff:
        for i, w_ff in enumerate(words_ff):
            if w_ff == se_ff:
                bag_ff[i] = 1

    return np.array(bag_ff)

def chatsa():
    count_sa = 0
    print("911 what is the nature of your emergency? Please choose from first aid, police, and firefighter.")
    print("In case of emergency, type emergency and we'll send help immediately. Type I’m fine, and we’ll end this and check back in on you. (type quit to exit)")
    print("1. First Aid bot")
    print("2. Police bot")
    print("3. Firefighter bot")
    print("4. EMERGENCY")

    while True:
        inp_sa = input("Type out your choice or enter the numbers (1 or 2 or 3 or 4): ")
        
        if inp_sa.lower() in ["quit"]:
            print("Have a nice day.")
            return
        if inp_sa.lower() in ["i'm fine", "im fine", "imfine"]:
            print("Have a nice day. We'll check in on you in a bit.")
            time.sleep(5)
            print("Just checking in on you again")
            chatcheckca()
            return
        if inp_sa.lower() in ["first aid", "firstaid", "1"]:
            chatfa()
            break
        elif inp_sa.lower() in ["police", "2"]:
            chatpo()
            break
        elif inp_sa.lower() in ["firefighter", "ff", "3"]:
            chatff()
            break
        elif inp_sa.lower() in ["emergency", "4"]:
            print("Please hang tight. Help is on the way")
            break
        else:
            count_sa += 1
            if count_sa == 3:
                print("We're not sure if you're pranking us. For your safety, police are on their way.")
                return
            else:
                print("Sorry, I couldn't understand that. Please type out your choice or enter the numbers (1 or 2 or 3 or 4):")
        
def chatfa():
    count_sa = 0
    print("What is your first aid emergency? (type quit to stop, and back to go back to the last menu)")

    while True:
        inp_fa = input("You: ")
        
        if inp_fa.lower() == "quit":
            break
        if inp_fa.lower() == "back":
            chatsa()
            break
        
        results_fa = model_fa.predict([bag_of_words_fa(inp_fa, words_fa)])
        results_index_fa = np.argmax(results_fa)
        tag_fa = labels_fa[results_index_fa]

        for tg_fa in data_fa["intents"]:
            if tg_fa['tag'] == tag_fa:
                responses_fa = tg_fa['responses']

        print(random.choice(responses_fa))

        count_sa += 1
        if count_sa == 3:
            print("We're not sure if you're pranking us. For your safety, police are on their way.")
            break 

def chatpo():
    count_sa = 0
    print("What is your police emergency? (type quit to stop, and back to go back to the last menu)")

    while True:
        inp_po = input("You: ")
        
        if inp_po.lower() == "quit":
            break
        if inp_po.lower() == "back":
            chatsa()
            break
        
        results_po = model_po.predict([bag_of_words_po(inp_po, words_po)])
        results_index_po = np.argmax(results_po)
        tag_po = labels_po[results_index_po]

        for tg_po in data_po["intents"]:
            if tg_po['tag'] == tag_po:
                responses_po = tg_po['responses']
                break

        else:
            count_sa += 1
            if count_sa == 3:
                print("We're not sure if you're pranking us. For your safety, police are on their way.")
                break

        print(random.choice(responses_po))
                
def chatff():
    count_sa = 0
    print("What is your firefighter emergency? (type quit to stop, and back to go back to the last menu)")

    while True:
        inp_ff = input("You: ")
        
        if inp_ff.lower() == "quit":
            break
        if inp_ff.lower() == "back":
            chatsa()
            break
        results_ff = model_ff.predict([bag_of_words_ff(inp_ff, words_ff)])
        results_index_ff = np.argmax(results_ff)
        tag_ff = labels_ff[results_index_ff]

        for tg_ff in data_ff["intents"]:
            if tg_ff['tag'] == tag_ff:
                responses_ff = tg_ff['responses']
                break

        else:
            count_sa += 1
            if count_sa == 3:
                print("We're not sure if you're pranking us. For your safety, police are on their way.")
                break

        print(random.choice(responses_ff))

def chatcheckca():
    count_ca = 0
    print("911 calling back to check in on you. If you're fine, please reply all is well. Type I'm fine and we'll send help immediately. (typing quit will not exit, this is a double safety feature.)")

    while True:
        inp_ch = input("You: ")

        if inp_ch.lower() in ["all is well", "alls well", "allswell", "all'swell"]:
            print("Have a nice day.")
            break
        if inp_ch.lower() in ["i'm fine", "im fine", "imfine"]:
            print("Have a nice day.")
            time.sleep(5)
            print("Units are enroute.")
            return
        else:
            count_ca += 1
            if count_ca == 3:
                print("We're not sure if you're pranking us. For your safety, police are on their way.")
                break
            print("Sorry, I couldn't understand that. Please try again.")
                    
    return

if __name__ == "__main__":
    chatsa()