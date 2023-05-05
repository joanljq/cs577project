import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''
from torch import nn
from torch import optim
from torch import tensor

import gensim.downloader as api

from neural_archs import DAN, RNN, LSTM
from utils import WiCDataset, sen2vec, sen2glove, sen2pos, sentence_tag

from nltk.data import load
tag_dict = load('help/tagsets/upenn_tagset.pickle')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # TODO: change the `default' attribute in the following 3 lines with the choice
    # that achieved the best performance for your case
    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='dan', type=str)
    parser.add_argument('--rnn_bidirect', default=False, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='glove', type=str)

    args = parser.parse_args()

    if args.init_word_embs == "glove":
        # TODO: Feed the GloVe embeddings to NN modules appropriately
        # for initializing the embeddings
        glove_embs = api.load("glove-wiki-gigaword-50")
        all_weights = glove_embs.get_normed_vectors()
        avg_wegihts = np.mean(all_weights,axis=0)
        update_weights = np.vstack((all_weights,avg_wegihts))
        weights = torch.FloatTensor(update_weights)
    else:
        # vocab size is 7459 based on experiment
        weights = torch.FloatTensor(np.random.rand(7459, 50))
		
    # TODO: Read off the WiC dataset files from the `WiC_dataset' directory
    # (will be located in /homes/cs577/WiC_dataset/(train, dev, test))
    # and initialize PyTorch dataloader appropriately
    # Take a look at this page
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # and implement a PyTorch Dataset class for the WiC dataset in
    # utils.py

    # root = '/homes/cs577/WiC_dataset/'
    # train_data = WiCDataset(root_dir=root)
    # vocab, _ = train_data.get_vocab()

    # test_data = WiCDataset('test',root_dir=root)
    # dev_data = WiCDataset('dev',root_dir=root)

    train_data = WiCDataset()
    vocab, _ = train_data.get_vocab()
    
    test_data = WiCDataset('test')
    dev_data = WiCDataset('dev')

    # ss = 'NN NNS NNP NNPS VB VBD VBG VBN VBP VBZ JJ JJR JJS RBR BR RBS WDT WP WP$ PRP PRP$ DT CD UH SYM FW  LS'
    # selected_tags = ss.split()
    # pos_dict = {}
    # i = 0
    # for pos in selected_tags:
    #     pos_dict[pos] = i
    #     i += 1
    # n = len(pos_dict)

    pos_dict = {}
    i = 0
    for pos in tag_dict.keys():
        pos_dict[pos] = i
        i += 1

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    pos_dict[START_TAG] = len(pos_dict)
    pos_dict[STOP_TAG] = len(pos_dict)

    # TODO: Freely modify the inputs to the declaration of each module below
    if args.neural_arch == "dan":
        model = DAN(args.init_word_embs,weights).to(torch_device)
    elif args.neural_arch == "rnn":
        if args.rnn_bidirect:
            model = RNN(True, args.init_word_embs,weights).to(torch_device)
        else:
            model = RNN(False, args.init_word_embs,weights).to(torch_device)
    elif args.neural_arch == "lstm":
        if args.rnn_bidirect:
            model = LSTM(True, args.init_word_embs, weights, pos_dict).to(torch_device)
        else:
            model = LSTM(False,args.init_word_embs, weights, pos_dict).to(torch_device)

    # TODO: Training and validation loop here

    print("programming is running")
    print(args)

    ce = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    patience = 5

    train_acc = []
    dev_acc = []
    test_acc = []

    train_loss = []
    train_label=[]
    train_crf=[]

    dev_loss = []

    test_loss = []

    best_val_loss = 1

    early_stop_counter = 0


    for epoch in range(epochs):

        model.train()

        #print("Epoch:",i)
        total_loss = 0
        total_label = 0
        total_crf = 0

        for i in range(len(train_data)):
            sample = train_data[i]
            
            optimizer.zero_grad()

            # a) calculate probs / get an output
            if args.init_word_embs == "glove":
                s1 = sen2glove(sample["sentence1"],glove_embs)
                s2 = sen2glove(sample["sentence2"],glove_embs)
            else:
                s1 = sen2vec(sample["sentence1"],vocab)
                s2 = sen2vec(sample["sentence2"],vocab)

            if args.neural_arch == "lstm":
                pos1 = torch.tensor(sentence_tag(sample['sentence1'], pos_dict), dtype=torch.long) # tensor([13, 35, 36, 17,  6, 11, 21])
                pos2 = torch.tensor(sentence_tag(sample['sentence2'], pos_dict), dtype=torch.long)
                y_raw,crfloss1,crfloss2,_,_ = model(s1,s2,pos1,pos2)
            else:
                y_raw = model(s1,s2)

            y = tensor(float(sample["label"]))
            
            # b) compute loss
            loss = ce(y_raw,y)*10+(crfloss1+crfloss2)/60
            total_loss += loss
            total_label+=ce(y_raw,y)*10
            total_crf+=(crfloss1+crfloss2)/60

            # c) get the gradient
            loss.backward()

            # d) update the weights
            optimizer.step()
        train_loss.append(total_loss.item()/len(train_data))
        train_label.append(total_label.item()/len(train_data))
        train_crf.append(total_crf.item()/len(train_data))
        print("---------epoch:",epoch,"---------")
        print(" train loss ", train_loss[-1]) 
        print(" train label ", train_label[-1]) 
        print(" train crf ", train_crf[-1])

        model.eval()

        score = 0
        for i in range(len(train_data)):
            sample = train_data[i]

            # a) calculate probs / get an output
            if args.init_word_embs == "glove":
                s1 = sen2glove(sample["sentence1"],glove_embs)
                s2 = sen2glove(sample["sentence2"],glove_embs)
            else:
                s1 = sen2vec(sample["sentence1"],vocab)
                s2 = sen2vec(sample["sentence2"],vocab)

            if args.neural_arch == "lstm":
                y_raw,crfloss1,crfloss2,_,_ = model(s1,s2)
            else:
                y_raw = model(s1,s2)

            result = True if y_raw >= 0.5 else False

            if bool(result) == sample["label"]:
                score += 1
        
        train_acc.append(score/len(train_data))

        print(" train accuracy ",train_acc[-1])

    # s1 = sen2glove(train_data[0]["sentence1"],glove_embs)
    # s2 = sen2glove(train_data[0]["sentence2"],glove_embs)
    # y_raw,_,_,pos_seq1,pos_seq2 = model(s1,s2)
    # pos1 = torch.tensor(sentence_tag(train_data[0]['sentence1'], pos_dict), dtype=torch.long) # tensor([13, 35, 36, 17,  6, 11, 21])
    # pos2 = torch.tensor(sentence_tag(train_data[0]['sentence2'], pos_dict), dtype=torch.long)
    # print(pos_seq1, pos1)
    # print(pos_seq2, pos2)


        score = 0
        total_loss = 0
        for i in range(len(dev_data)):
            sample = dev_data[i]
            # a) calculate probs / get an output
            if args.init_word_embs == "glove":
                s1 = sen2glove(sample["sentence1"],glove_embs)
                s2 = sen2glove(sample["sentence2"],glove_embs)
            else:
                s1 = sen2vec(sample["sentence1"],vocab)
                s2 = sen2vec(sample["sentence2"],vocab)

            if args.neural_arch == "lstm":
                y_raw,crfloss1,crfloss2,_,_ = model(s1,s2)
            else:
                y_raw = model(s1,s2)

            y = tensor(float(sample["label"]))
            loss = ce(y_raw,y)*10
            total_loss += loss

            result = True if y_raw >= 0.5 else False
            if bool(result) == sample["label"]:
                score += 1

        dev_acc.append(score/len(dev_data))
        print(" val accuracy ",dev_acc[-1])

        dev_loss.append(total_loss.item()/len(dev_data))
        print(" dev loss ", dev_loss[-1])

    # TODO: Testing loop
    # Write predictions (F or T) for each test example into test.pred.txt
    # One line per each example, in the same order as test.data.txt.
        test_output = []
        score = 0
        total_loss = 0
        for i in range(len(test_data)):
            sample = test_data[i]
            # a) calculate probs / get an output
            if args.init_word_embs == "glove":
                s1 = sen2glove(sample["sentence1"],glove_embs)
                s2 = sen2glove(sample["sentence2"],glove_embs)
            else:
                s1 = sen2vec(sample["sentence1"],vocab)
                s2 = sen2vec(sample["sentence2"],vocab)

            if args.neural_arch == "lstm":
                y_raw,crfloss1,crfloss2,_,_ = model(s1,s2)
            else:
                y_raw = model(s1,s2)
        
            y = tensor(float(sample["label"]))
            loss = ce(y_raw,y)*10
            total_loss += loss

            result = True if y_raw >= 0.5 else False
            if bool(result) == sample["label"]:
                score += 1

            output = "T" if y_raw >= 0.5 else "F"
            test_output.append(output)

        test_acc.append(score/len(test_data))
        print(" test accuracy ",test_acc[-1])

        test_loss.append(total_loss.item()/len(test_data))
        print(" test loss ", test_loss[-1])


        if epoch% 10 == 0:
            print("---------epoch:",epoch,"---------")
            print(" train loss ", train_loss[-1]) 
            print(" dev loss ", dev_loss[-1])
            print(" test loss ", test_loss[-1]) 
            print(" train accuracy ",train_acc[-1])
            print(" dev accuracy ",dev_acc[-1])
            print(" test accuracy ",test_acc[-1])

        # check if validation loss has improved val loss < best val loss:
        if dev_loss[-1] < best_val_loss:
            best_val_loss = dev_loss[-1]
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("------Early stopping after epoch:", epoch, "---------")
                print(" train loss ", train_loss[-1])
                print(" val loss ", dev_loss[-1])
                print(" test loss ", test_loss[-1])
                print(" train accuracy ", train_acc[-1])
                print(" val accuracy ", dev_acc[-1])
                print(" test accuracy ", test_acc[-1])
                break


    with open('test.pred.txt', 'w') as f:
        for line in test_output:
            f.write(f"{line}\n")

    print("programming ends")


    # cross validation code using DAN as an example
    # def DAN_kFold(k, epochs,lr0,*inputs):
    
    #     num_val_samples = len(train_data)//k
    #     cv_score = []
    #     for i in range(k):
    #         print('Processing fold: ', i + 1)
    #         """%%%% Initiate new model %%%%""" #in every fold
    #         model = DAN(*inputs).to(torch_device)
            
    #         valid_idx = np.arange(len(train_data))[i * num_val_samples:(i + 1) * num_val_samples]
    #         train_idx = np.concatenate([np.arange(len(train_data))[:i * num_val_samples], np.arange(len(train_data))[(i + 1) * num_val_samples:]], axis=0)
            
    #         train_dataset = Subset(train_data, train_idx)
    #         valid_dataset = Subset(train_data, valid_idx)

            
    #         _,_,valid_acc = training(model,train_dataset, valid_dataset,lr0,epochs)
    #         cv_score.append(valid_acc[-1])
        
    #     print('cv_score: ',sum(cv_score)/len(cv_score))

    #     return sum(cv_score)/len(cv_score)
    

    # def training(model,train_dataset, valid_dataset,lr0,epochs):
    #     ce = nn.BCELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=lr0)

    #     train_acc = []
    #     dev_acc = []
    #     train_loss = []
    #     val_loss = []
    #     # test_output = []

    #     best_val_loss = 1
    #     early_stop_counter=0
        
    #     for epoch in range(epochs):

    #         model.train()

    #         #print("Epoch:",i)
    #         total_loss = 0

    #         for i in range(len(train_dataset)):
    #             sample = train_dataset[i]
                
    #             optimizer.zero_grad()

    #             # a) calculate probs / get an output
    #             if init_word_embs == "glove":
    #                 s1 = sen2glove(sample["sentence1"],glove_embs)
    #                 s2 = sen2glove(sample["sentence2"],glove_embs)
    #             else:
    #                 s1 = sen2vec(sample["sentence1"])
    #                 s2 = sen2vec(sample["sentence2"])

    #             y_raw = model(s1,s2)

    #             y = tensor(float(sample["label"]))
                
    #             # b) compute loss
    #             loss = ce(y_raw,y)
    #             total_loss += loss

    #             # c) get the gradient
    #             loss.backward()

    #             # d) update the weights
    #             optimizer.step()
    #         train_loss.append(total_loss.item()/len(train_dataset))

    #         model.eval()

    #         score = 0
            
    #         for i in range(len(train_dataset)):
    #             sample = train_dataset[i]

    #             # a) calculate probs / get an output
    #             if init_word_embs == "glove":
    #                 s1 = sen2glove(sample["sentence1"],glove_embs)
    #                 s2 = sen2glove(sample["sentence2"],glove_embs)
    #             else:
    #                 s1 = sen2vec(sample["sentence1"])
    #                 s2 = sen2vec(sample["sentence2"])

    #             y_raw = model(s1,s2)

    #             result = True if y_raw >= 0.5 else False

    #             if bool(result) == sample["label"]:
    #                 score += 1
            
    #         train_acc.append(score/len(train_dataset))
            

    #         score = 0
    #         total_loss = 0
    #         for i in range(len(valid_dataset)):
    #             sample = valid_dataset[i]
    #             # a) calculate probs / get an output
    #             if init_word_embs == "glove":
    #                 s1 = sen2glove(sample["sentence1"],glove_embs)
    #                 s2 = sen2glove(sample["sentence2"],glove_embs)
    #             else:
    #                 s1 = sen2vec(sample["sentence1"])
    #                 s2 = sen2vec(sample["sentence2"])

    #             y_raw = model(s1,s2)
                
    #             y = tensor(float(sample["label"]))
    #             loss = ce(y_raw,y)
    #             total_loss += loss

    #             result = True if y_raw >= 0.5 else False
    #             if bool(result) == sample["label"]:
    #                 score += 1

    #         val_loss.append(total_loss.item()/len(valid_dataset))
    #         dev_acc.append(score/len(valid_dataset))



    #         if epoch% 10 == 0:
    #             print("---------epoch:",epoch,"---------")
    #             print(" train loss ", train_loss[-1]) 
    #             print(" val loss ", val_loss[-1])
    #             print(" train accuracy ",train_acc[-1])
    #             print(" val accuracy ",dev_acc[-1])

    #         # check if validation loss has improvedval loss < best val loss:
    #         if val_loss[-1] < best_val_loss:
    #             best_val_loss = val_loss[-1] 
    #             early_stop_counter = 0
    #         else:
    #             early_stop_counter += 1
    #             if early_stop_counter >= patience:
    #                 print("------Early stopping after epoch:",epoch,"---------")
    #                 print(" train loss ", train_loss[-1]) 
    #                 print(" val loss ", val_loss[-1]) 
    #                 print(" train accuracy ",train_acc[-1])
    #                 print(" val accuracy ",dev_acc[-1])
    #                 return train_loss,train_acc,dev_acc


    #     print("---------endng epoch:",epoch,"---------")
    #     print(" train loss ", train_loss[-1]) 
    #     print(" val loss ", val_loss[-1]) 
    #     print(" train accuracy ",train_acc[-1])
    #     print(" val accuracy ",dev_acc[-1])
    #     return train_loss,train_acc,dev_acc



    # init_word_embs = 'glove'
    # # neural_arch = 'dan'
    # # rnn_bidirect = False

    # if init_word_embs == "glove":
    # # TODO: Feed the GloVe embeddings to NN modules appropriately
    # # for initializing the embeddings
    # glove_embs = api.load("glove-wiki-gigaword-50")
    # all_weights = glove_embs.get_normed_vectors()
    # avg_wegihts = np.mean(all_weights,axis=0)
    # update_weights = np.vstack((all_weights,avg_wegihts))
    # weights = torch.FloatTensor(update_weights)
    # else:
    # # vocab size is 7459 based on experiment
    # weights = torch.FloatTensor(np.random.rand(7459, 50))

    # print(init_word_embs,'weights')

    # epochs = 50
    # patience = 5
    # k = 5

    # params = {}
    # params['lr0'] = [0.00001,0.0001, 0.001,0.01,0.1]

    # params['hidden_dim'] = [250]
    # params['p_drop'] = [0]

    # result = []

    # best_params = {}
    # best_score = 0
    # for lr0 in params['lr0']:
    # for hidden_dim in params['hidden_dim']:
    #     for p_drop in params['p_drop']:
    #         inputs = init_word_embs, weights, hidden_dim, p_drop
    #         score = DAN_kFold(k, epochs,lr0,*inputs)
    #         result.append(score)

    #         print('current setting is ','lr0',lr0,'layer_num 1','hidden_dim',hidden_dim,'p_drop',p_drop)
    #         print('current score is',score)

    #         if score>best_score:
    #             best_score = score
    #             best_params['lr0'] = lr0
    #             best_params['layer_num'] = 1
    #             best_params['hidden_dim'] = hidden_dim
    #             best_params['p_drop'] = p_drop

    # print('best score is', best_score)
    # print('best_parameters are', best_params)

