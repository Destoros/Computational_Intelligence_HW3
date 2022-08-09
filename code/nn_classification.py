from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_boxplot, plot_image
import numpy as np

import IPython

"""
Assignment 3: Neural networks
Part 2: Classification with Neural Networks: Fashion MNIST

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def ex_2_1(X_train, y_train, X_test, y_test):
    """
    Solution for exercise 2.1
    :param X_train: Train set
    :param y_train: Targets for the train set
    :param X_test: Test set
    :param y_test: Targets for the test set
    :return:
    """
    #MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000
    
    n_hidd = 10;
    seeds = range(5)
    
    train_acc = np.zeros((len(seeds),))
    test_acc = np.zeros((len(seeds),))
    
    for seed in seeds:
        clf = MLPClassifier(hidden_layer_sizes = (n_hidd,),activation='tanh',max_iter=50, random_state=seed).fit(X_train, y_train).fit(X_train, y_train)        
        
        train_acc[seed] = clf.score(X_train,y_train)
        test_acc[seed] = clf.score(X_test,y_test)
        

        
    IPython.embed()     
    plot_boxplot(train_acc, test_acc)
    
    best_seed = np.argmax(test_acc)
    
    #train the calssifier with the best seed 
    clf = MLPClassifier(hidden_layer_sizes = (n_hidd,),activation='tanh',max_iter=50, random_state=best_seed).fit(X_train, y_train).fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    C = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None)
    print(C)
    
    hidden_layer_weights = clf.coefs_[0] #weights from to 0-th layer to the 1-st layer (input layer to frist hidden layer)

    plot_hidden_layer_weights(hidden_layer_weights)
    
    
    flag = 0
    list_objects = ("T-shirt/top", "trousers/pants", "pullover shirt", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")   
    for ii in range(len(X_train)):
        
        score = clf.score( X_train[ii:ii+1],y_train[ii:ii+1] )
        
        if score < 1E-12: #score delivers float
            #enters if misclassified
            print("Should be a", list_objects[y_train[ii]], "and was classified as", list_objects[int(clf.predict(X_train[ii:ii+1]))] )
            plot_image(X_train[ii:ii+1].reshape(28,28))       
            flag += 1
           
        if flag == 5: #after 5 missclassified images break
            break


    #Include the confusion matrix you obtain and discuss. Are there any clothing items which can be better separated than others?
    #confusion_matrix =   [[836   3  16  50   4   1  76   0  14   0]
                         # [  3 947   9  35   3   0   2   0   1   0]
                         # [ 16   4 783  12 110   2  64   0   9   0]
                         # [ 27  10  15 882  29   2  24   0  11   0]
                         # [  0   1 101  41 786   0  67   0   4   0]
                         # [  1   0   1   0   0 936   0  33   5  24]
                         # [161   3 113  52  88   0 561   1  21   0]
                         # [  0   0   0   0   0  29   0 929   0  42]
                         # [  3   1   4   9   5   6  12   5 955   0]
                         # [  1   0   0   0   0  10   0  38   1 950]]
        #The confusion matrix display how often an object has been classified as which object. The first row displays how of the first object has been classified as j-th object, where j also denotes the column. e.g the first object has been classified as the second object 3 times. The main diagonal shows often the objects were classified correctly(for our data it would be ideally 1000).
        
        # The second(trousers/pants) and last two objects ("bag" and "ankle boot") can be detectet very well, which is displayed by the large number in the main diagonal. 
        
        #An example for badly recognized object would be the row nr 7 ("shirt"), because its main diagonal number is quite low. It was often mistaken for object 1 ("T-shirt/top"), object 3 ("pullover shirt")and object 5 ("coat").
                      
  
    #Can you ﬁnd particular regions of the images which get more weights than others?
    # NOCH BEANTWORTEN
    
    #Explain the boxplot
    #Since we only calculate 5 points of accuracy, each line of the boxplot correspond to a calculated accuracy value 
  

    ## TODO
    pass
