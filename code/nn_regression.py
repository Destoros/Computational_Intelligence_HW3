import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

import IPython


from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, \
    plot_learned_function, plot_mse_vs_alpha

"""
Assignment 3: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def calculate_mse(nn, x, y):
    """
    Calculates the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    y_pred = nn.predict(x)
    mse = np.mean((y - y_pred)**2)
    

    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    #MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

    n_hidden_layer = (2,5,50) 
    #we expect: underfitting for 2, good fit for 5 and overfitting for 50
    
    for n_h in n_hidden_layer:
        regr = MLPRegressor(hidden_layer_sizes = (n_h,), solver='lbfgs',activation='logistic', alpha=0, max_iter=5000).fit(x_train, y_train)
        
        y_pred_train = regr.predict(x_train)
        y_pred_test = regr.predict(x_test)
        

        plot_learned_function(n_h, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)

    ## TODO
    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    hidden_layer = (5,)
    MSE_train = []
    MSE_test = []
    for counter in range(10):
    
        regr = MLPRegressor(hidden_layer_sizes = hidden_layer, solver='lbfgs',activation='logistic',random_state=counter, alpha=0, max_iter=5000).fit(x_train, y_train)
        
        y_pred_train = regr.predict(x_train)
        y_pred_test = regr.predict(x_test)
        
        n_hidden = hidden_layer[0]

        
        
        MSE_train.append(calculate_mse(regr, x_train, y_train))
        MSE_test.append(calculate_mse(regr, x_test, y_test))
        print("MSE_train for state =", counter, ":", MSE_train[-1])
        print("MSE_test:", MSE_test[-1])
        print("\n")
        
        # plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
        
 
    mean = np.mean(MSE_train)
    min = np.min(MSE_train)
    max = np.max(MSE_train)
    std = np.std(MSE_train)
    
    print("mean =", mean)
    print("min =", min)
    print("max =", max)
    print("std =", std)
    #What is the minimum, maximum, mean and standard deviation of the mean squared error obtained on the training set?
    #see prot folder
    
    # Is the minimum MSE obtained for the same seed on the training and on the testing set?
    # 2BP: No, the minimum MSE is not on the same seed 
    
    #Explain why you would need a validation set to choose the best seed
    #as far as i understand we want to know how the nn behaves to data which it didnt see before. If we use the
    #test set to find the perfect model degree, we change the nn in such a way, that it fits to the test set well. Therefore
    # the test set is implicitly seen from the nn and we cant check how it behaves on new data anymore. (as long as we dont get some new data somehow)
    
    #Unlike linear-regression and logistic regression, even if the algorithm converged, the variability of the MSE across seeds is expected. Why
    #Because we dont have a convex cost function anymore. Depending on the inital weights we may converged to another valley each time of the cost function.
    
    #What is the source of randomness introduced by Stochastic Gradient Descent (SGD)? What source of randomness will persist if SGD is replaced by standard Gradient Descent?
    #The standard GD uses the entire data set to calc the 'downhill' direction. SGD only uses only 1 sample, decreasing the amount of calculation a lot. But this causes the calculated gradient to not point to the steepest downhill direction, it is a bit off. 
    
    #??NOT SURE:?? Even if SGD is replaced by standard GD, the valley in which we end up depends on the random initializied weights of the nn.
    
    ## TODO
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """


    n_h = [1,2,4,6,8,12,20,40] #hidden neurons in one layer
    n_rnd_state = 10
    MSE_train = np.zeros((len(n_h),n_rnd_state))
    MSE_test = np.zeros((len(n_h),n_rnd_state))
    
    for index,n_hidden in enumerate(n_h):
        
        for counter in range(n_rnd_state):
              
            regr = MLPRegressor(hidden_layer_sizes = (n_hidden,), solver='lbfgs',activation='logistic',random_state=counter, alpha=0, max_iter=5000).fit(x_train, y_train)
            
            y_pred_train = regr.predict(x_train)
            y_pred_test = regr.predict(x_test)
            
            # plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
                 
            MSE_train[index,counter] = calculate_mse(regr, x_train, y_train)
            MSE_test[index,counter]  = calculate_mse(regr, x_test, y_test)
            
            
            
    # print("MSE train:", MSE_train)
    # print("MSE test:", MSE_test)

   
    #Beim durchgehen einzelnen plots mit plot_learned_function() fiel auf, dass überwiegend gute ergebnisse lieferte, aber bei manche teilen die limits des plots sprengte, wodurch der der MSE zu den vielen test samples groß wird und durch die aufsummierung dieser, den plot demenstprechend aussehen lässt. Könnte durch np.median() gelöst werden .
        
   
    plot_mse_vs_neurons(MSE_train, MSE_test, n_h)
    i_best = np.argmin(MSE_test.mean(axis=1)) #only use the set, because the training set would get better with each neuron

    
    n_h_best = n_h[i_best]
    print("n_h_best = ", n_h_best) #our example 8, gilt nur unter der annahme, dass wir keine einfluss auf die seeds haben
    
    regr = MLPRegressor(hidden_layer_sizes = (n_h_best,), solver='lbfgs',activation='logistic', alpha=0, max_iter=5000).fit(x_train, y_train)
    
    y_pred_train = regr.predict(x_train)
    y_pred_test = regr.predict(x_test)   
            
    plot_learned_function(n_h_best, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
    
    

    

    #What is the best value of nh independent of the choice of the random seed? Use errors that are averaged over runs with diﬀerent random seeds.
    #This would only be possible if we average over a nearly infinte amount of rnd seeds. We compared two different seed ranges(from 0 to 9 and from 10 to 20), which resultet in two different n_h_best. 
    
    #over- and underfitting: same as before
    ## TODO
    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 d)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    
    n_h = [2,5,50]
    n_iterations = 5000
    rnd_state = 0

    # (len(hidden_neuron_list),n_iterations)
    MSE_test = np.zeros((len(n_h),n_iterations))
    MSE_train = np.zeros((len(n_h),n_iterations))
  
    
    for index,n_hidden in enumerate(n_h):        
        #                                                   DONT KNOW IF SOLVER SHOULD BE LBFGS OR SGD; WAS NOT STATED IN ASSIGNMENT AND OF THE Q's ASKED FOR SGD
        regr = MLPRegressor(hidden_layer_sizes = (n_hidden,), warm_start=True, solver='lbfgs',activation='logistic',random_state=rnd_state, alpha=0, max_iter=1)  
        
        for counter in range(n_iterations): 
            regr.fit(x_train, y_train)
            
            MSE_train[index,counter] = calculate_mse(regr, x_train, y_train)
            MSE_test[index,counter] = calculate_mse(regr, x_test, y_test)
            

            # if counter % 1000 == 0 and n_hidden == 50:
                # y_pred_train = regr.predict(x_train)
                # y_pred_test = regr.predict(x_test)   
                        
                # plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
    
            # print(MSE_train)
            
            # HIiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii<333
        
        

    plot_mse_vs_iterations(MSE_train, MSE_test, n_iterations, n_h)
    
    #Does the risk of overﬁtting increase or decrease with the number of hidden neurons?
    #Assuming we are overfitting in the first place, the risk of overfitting increases with the amount of hidden neurons.
    
    #ASSUMPTION: QUESTION BEFORE WAS STATED WRONG
    # Does the risk of overﬁtting increase or decrease with the number iterations?
    #For the training it gets better with increased number of iterations. The MSE of the test set varies. It increases and decreases over iterations with no overall trend.
    
    #What feature of stochastic gradient descent helps to overcome overﬁtting?
    #in own words: SGD only uses one data sample to calculate the gradient. Hence it is not able to find the true gradient which fits a curve to all given samples and therefore causing oscillations or extreme peaks(when overfitting). SGD only changes to curve in such a way that the new model fits better for the one sample it used to calculated the gradient. ???????????????????????????????????????????????????????????????????????????????
    
    #From Inannet:
    #The reason for which SGD prevents overfitting by design is, once again, given by the limited subset of data points used to train the model. If SGD overfits the training data in a number of iterations, it is still guaranteed to generalize because that training subset is so small that overfitting would not be critical. Of course, those who would like a formal proof of what has been claimed thus far, need to read the paper, which might be a bit challenging but definitely interesting.
    
    #THINK: WHY SHOULD IT MATTER HOW WE GOT TO THE BOTTOM OF THE VALLEY; THE LOCAL MINIMUM SHOULD STILL DELIVER THE SAME WEIGHTS NO MATTER THE PATH THERE
    
    ## TODO
    pass


def ex_1_2(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    n_hidden = 50
    
    alphas = [10E-6,10E-5,10E-4,10E-3,10E-2,10E-1,1,10,100]
    seeds = range(10) 
    
    MSE_train = np.zeros( (len(alphas), len(seeds)) )
    MSE_test  = np.zeros( (len(alphas), len(seeds)) )
    
    for ii,alpha_it in enumerate(alphas):
        for jj,seed in enumerate(seeds):
            
            regr = MLPRegressor(hidden_layer_sizes = (n_hidden,), solver='lbfgs',activation='logistic',random_state=seed, alpha=alpha_it, max_iter=5000).fit(x_train, y_train)

            MSE_train[ii,jj] = calculate_mse(regr, x_train, y_train)
            MSE_test[ii,jj] = calculate_mse(regr, x_test, y_test)
            
            
        # y_pred_train = regr.predict(x_train)
        # y_pred_test = regr.predict(x_test)
        
        # plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
            
       
    plot_mse_vs_alpha(MSE_train, MSE_test, alphas)
    ## TODO
    #What is/are the best value(s) of α?
        # 10E-3
        
    #Is regularization used to overcome overﬁtting or underﬁtting? Why?
        # overﬁtting, because it punishes complexe models (since the weight matrix W contains more and higher values -> ||W|| increases -> MSE_reg increases)
        
        
    pass
    
    
    
  
    
    
    
