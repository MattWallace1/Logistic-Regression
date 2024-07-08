import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

logistic = lambda u: 1/(1+np.exp(-u))

def train(N=200, alpha=1e-3):
    reviews_positive = []
    reviews_negative = []
    Ws = {}
    b = 0
    
    # set up vocabulary and initialize weights as 0.5 since
    # all weights will be 0.5 after first epoch
    for i in range(1000):
        fin = open("movies/pos/{}.txt".format(i), encoding="utf8")
        fin.close()
        fin = open("movies/neg/{}.txt".format(i), encoding="utf8")
        words_negative = set(fin.read().lower().split())
        reviews_negative.append(words_negative)
        words_positive = set(fin.read().lower().split())
        reviews_positive.append(words_positive)
        for word in words_positive:
            Ws[word] = 0
        for word in words_negative:
            Ws[word] = 0
        
    print("model built")
    all_reviews = [reviews_negative, reviews_positive]
    dWs = {key:0 for key in Ws}
    db = 0
    yi = 0
    for epoch in range(N):
        ipd.clear_output()
        print("{}/{}".format(epoch+1, N))
        for review_set in all_reviews:
            for review in review_set:
                u = 0
                for word in review:
                    u += Ws[word]
                y_est = logistic(u)
                for word in review:
                    dWs[word] += yi - y_est
                db += yi - y_est
        Ws = {key:Ws[key] + alpha * dWs[key] for key in Ws}
        b += b + alpha * db
        yi += 1 # switching to positive reviews in next set
    
    return Ws

def cross_validate(n_folds=5, N=200, alpha=1e-3):
    
    reviews = []
    ys = [] # store labels in parallel array
    for i in range(1000):
        fin = open("movies/pos/{}.txt".format(i), encoding="utf8")
        reviews.append(fin.read().lower().split())
        ys.append(1)
        fin.close()
        fin = open("movies/neg/{}.txt".format(i), encoding="utf8") 
        reviews.append(fin.read().lower().split())
        ys.append(0)
        fin.close()
        
    # store all reviews up front, randomly permute
    # separate into folds using np.split()
    reviews = np.array(reviews)
    ys = np.array(ys)
    permute_idxs = np.random.permutation(len(reviews))
    reviews = reviews[permute_idxs]
    ys = ys[permute_idxs]
    folds = np.split(reviews, n_folds)
    ys = np.split(ys, n_folds)
    
    accuracies = []
    for fold_num, test_set in enumerate(folds):
        Ws = {}
        b = 0
        
        # takes the set difference of reviews minus test_set
        training_set = np.setdiff1d(reviews, test_set)
        
        # initialize bag of words
        for review in training_set:
            for word in review:
                Ws[word] = 0
                
        # train
        dWs = {key:0 for key in Ws}
        db = 0
        for epoch in range(N):
            ipd.clear_output()
            print("Fold {}: {}/{}".format(fold_num, epoch+1, N))
            for review in training_set:
                yi = review[1]
                u = 0
                for word in review[0]:
                    u += Ws[word]
                y_est = logistic(u)
                for word in review[0]:
                    dWs[word] += yi - y_est
                db += yi - y_est
            Ws = {key:Ws[key] + alpha * dWs[key] for key in Ws}
            b += b + alpha * db
            
        # test
        num_correct = 0
        for review in test_set:
            yi = review[1]
            u = 0
            for word in review[0]:
                if word in Ws:
                    u += Ws[word]
            y_est = logistic(u)
            if np.abs(yi - y_est) < 0.5:
                num_correct += 1
        accuracies.append(num_correct/len(test_set))
            
    return accuracies     