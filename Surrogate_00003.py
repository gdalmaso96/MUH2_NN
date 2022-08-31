import optuna
import time
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.models import load_model
from multiprocessing.pool import ThreadPool
import subprocess

np.random.seed(299792458)
set_random_seed(299792458)

######### Function DBtonumpy: extracts DB data to be saved in npy files
def DBtonumpy(studyname, database, objective, normalization):
    # Load study
    study = optuna.load_study(study_name=studyname, storage=database)
    
    trials_all = study.get_trials()
    
    # Shuffle data
    np.random.shuffle(trials_all)
    
    trials_p = []
    trials_v = []
    
    for trial in trials_all:
        if(trial.state==optuna.trial.TrialState(1) and trial.values[0] > 0 and trial.values[3] < 80):
            params = trial.params
            params.pop("halfgapL")
            params.pop("halfgapR")
            trials_p.append(params)
            V = []
            for o in objective:
                V.append(trial.values[int(o)])
            trials_v.append(V)
    
    # Convert data to numpy arrays and rescale between 0 and 1
    X = np.zeros((len(trials_p), len(list(trials_p[0]))))
    Y = np.zeros((len(trials_p), len(objective)))
    
    for i in range(len(trials_p)):
        pars = trials_p[i]
        for j in range(len(list(pars))):
            X[i][j] = pars[list(pars)[j]]
        vals = trials_v[i]
        
        for j in range(len(normalization)):
            Y[i][j] = vals[j]/normalization[j]
        
    np.save("X.npy", X)
    np.save("Y.npy", Y)



######### Function train: as an input it takes the objective index, and the DB name. It saves the resulting model and plots the learning results
def train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs):
    # Load data
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    
    # Save data batch for test
    X_train, X_test = np.split(X, [int(len(X)*0.8)])
    Y_train, Y_test = np.split(Y, [int(len(X)*0.8)])
    
    # Check if model exists
    if not os.path.exists(studyname + '_%03dneurons_%03dlayers_%04depochs_obj%d_%dbatch_save' %(nnodes, depth, epochs, len(objective), batchsize)):
        # Create model
        model = Sequential()
        model.add(Dense(nnodes, input_shape=(len(list(X[0])), ), activation='tanh'))
        for i in range(depth - 1):
            model.add(Dense(nnodes, activation='tanh'))
        
        model.add(Dense(len(objective)))
    else:
        model = load_model(studyname + '_%03dneurons_%03dlayers_%04depochs_obj%d_%dbatch_save' %(nnodes, depth, epochs, len(objective), batchsize))
    
    # Train
    optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimiser, loss='mse')
    
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batchsize, validation_split=0.25, shuffle=True) #add validate
    
    results = model.evaluate(X_test, Y_test)
    

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.yscale('log')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig('fig/' + studyname + '_%03dneurons_%03dlayers_%04depochs_obj%d_%dbatch_train_loss.png' %(nnodes, depth, epochs, len(objective), batchsize))
    plt.clf()
    
            
    # Save model
    model.save(studyname + '_%03dneurons_%03dlayers_%04depochs_obj%d_%dbatch_save' %(nnodes, depth, epochs, len(objective), batchsize))
    
    # Test correlation
    corr = []
    
    for i in range(len(normalization)):
        Y_pred = model.predict(X_train)
        plt.plot(Y_pred[:, i], Y_train[:, i], '.', label=r'$\rho$ = %.6f' %(np.corrcoef(Y_pred[:, i], Y_train[:, i])[0][1]))
        if(objective[i] == 0):
            plt.plot([0, 0.2], [0, 0.2], '--', color = 'r')
        else:
            plt.plot([0, 1], [0, 1], '--', color = 'r')
        
        plt.title('Prediction on train, objective %d' %(objective[i]))
        plt.ylabel('Train value')
        plt.xlabel('Prediction')
        plt.legend(loc='upper left')
        #plt.show()
        plt.savefig('fig/' + studyname + '_%03dneurons_%03dlayers_%04depochs_%dobj%d_%dbatch_train.png' %(nnodes, depth, epochs, objective[i], len(objective), batchsize))
        plt.clf()
        
        # Create test plot
        Y_pred = model.predict(X_test)
        
        plt.plot(Y_pred[:, i], Y_test[:, i], '.', label=r'$\rho$ = %.6f' %(np.corrcoef(Y_pred[:, i], Y_test[:, i])[0][1]))
        if(objective[i] == 0):
            plt.plot([0, 0.2], [0, 0.2], '--', color = 'r')
        else:
            plt.plot([0, 1], [0, 1], '--', color = 'r')
        
        plt.title('Prediction on test, objective %d' %(objective[i]))
        plt.ylabel('Test value')
        plt.xlabel('Prediction')
        plt.legend(loc='upper left')
        #plt.show()
        plt.savefig('fig/' + studyname + '_%03dneurons_%03dlayers_%04depochs_%dobj%d_%dbatch_test.png' %(nnodes, depth, epochs, objective[i]))
        plt.clf()
        
        corr.append(np.corrcoef(Y_pred[:, i], Y_test[:, i])[0][1])
    
    
    
    return results

if __name__ == '__main__':
    studyname = 'MUH2_V5_Peter_TOT_mup_mono'
    database = 'sqlite:///' + studyname + '.db'
    objective = []
    objective.append(0)
    '''
    objective.append(1)
    objective.append(2)
    objective.append(3)
    objective.append(4)
    '''
    batchsize_s = 100
    normalization = []
    normalization.append((2922072/2e11*2.4e-3/1.6e-19)*0.2) # 20% of total muons
    '''
    normalization.append(200) # 200 mm
    normalization.append(200) # 200 mm
    normalization.append(100) # 100 MeV/c
    normalization.append(100) # 100 %
    '''
    nnodes_s = 20
    depth_s = 1
    epochs = 10000
    start_time = time.time()
    #DBtonumpy(studyname, database, objective, normalization)
    for i in range(20):
            for k in range(10):
            depth = depth_s + i
            nnodes = nnodes_s
            batchsize = batchsize_s + k*100
            train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 1 layer
            print(time.time() - start_time)
            
        
    
