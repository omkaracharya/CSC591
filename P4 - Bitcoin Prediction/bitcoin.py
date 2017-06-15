import statsmodels.formula.api as smf
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import math
import sys


#Path to the dataset
if len(sys.argv) != 2:
    print('bitcoin.py <path to data folder>')
    sys.exit(1)
data_path = sys.argv[1]


# Reading the vectors from the given csv files
train1_90 = pd.read_csv(data_path+'/train1_90.csv')
train1_180 = pd.read_csv(data_path+'/train1_180.csv')
train1_360 = pd.read_csv(data_path+'/train1_360.csv')

train2_90 = pd.read_csv(data_path+'/train2_90.csv')
train2_180 = pd.read_csv(data_path+'/train2_180.csv')
train2_360 = pd.read_csv(data_path+'/train2_360.csv')

test_90 = pd.read_csv(data_path+'/test_90.csv')
test_180 = pd.read_csv(data_path+'/test_180.csv')
test_360 = pd.read_csv(data_path+'/test_360.csv')

#Calculate similarity between two objects
def similarity(X, Xii):
    X = X[:-1]
    Xii = Xii[:-1]

    M = len(X)
    
    Xmean = np.mean(X)
    Xstd = np.std(X)
    
    Xiimean = np.mean(Xii)
    Xiistd = np.std(Xii)
    
    sim = 0
    for z in range(M):
       sim = sim + (X[z] -  Xmean) * (Xii[z] - Xiimean)
    
    sim = sim / (M * Xstd * Xiistd)
    return sim


def computeDelta(wt, X, Xi):
    """
    Parameters
    ----------
    wt : int
        This is the constant c at the top of the right column on page 4.
    X : A row of Panda Dataframe
        Corresponds to (x, y) in Equation 6.
    Xi : Panda Dataframe
        Corresponds to a dataframe of (xi, yi) in Equation 6.

    Returns
    -------
    float
        The output of equation 6, a prediction of the average price change.
    """
    
    numerator = 0
    denominator = 0 
    
    for i in range(len(Xi)):
        numerator = numerator + (Xi.iloc[i][-1] * math.exp(wt * similarity(X, Xi.loc[i])))
        denominator = denominator + math.exp(wt * similarity(X, Xi.loc[i]))
        
    Eemp = numerator / denominator
    return Eemp
        
# Bayesian Regression to predict the average price change for each dataset of train2 using train1 as input. 
# These will be used to estimate the coefficients (w0, w1, w2, and w3) in equation 8.
weight = 2
trainDeltaP90 = np.empty(0)
trainDeltaP180 = np.empty(0)
trainDeltaP360 = np.empty(0)
for i in xrange(0,len(train1_90.index)) :
    trainDeltaP90 = np.append(trainDeltaP90, computeDelta(weight,train2_90.iloc[i],train1_90))
for i in xrange(0,len(train1_180.index)) :
    trainDeltaP180 = np.append(trainDeltaP180, computeDelta(weight,train2_180.iloc[i],train1_180))
for i in xrange(0,len(train1_360.index)) :
    trainDeltaP360 = np.append(trainDeltaP360, computeDelta(weight,train2_360.iloc[i],train1_360))

# Actual deltaP values for the train2 data.
trainDeltaP = np.asarray(train2_360[['Yi']])
trainDeltaP = np.reshape(trainDeltaP, -1)

# Combine all the training data
d = {'deltaP': trainDeltaP,
     'deltaP90': trainDeltaP90,
     'deltaP180': trainDeltaP180,
     'deltaP360': trainDeltaP360 }
trainData = pd.DataFrame(d)

model = smf.ols('deltaP ~ deltaP90 + deltaP180 + deltaP360', trainData) 
model = model.fit()

# Print the weights from the model
print model.params

testDeltaP90 = np.empty(0)
testDeltaP180 = np.empty(0)
testDeltaP360 = np.empty(0)
for i in xrange(0,len(train1_90.index)) :
    testDeltaP90 = np.append(testDeltaP90, computeDelta(weight,test_90.iloc[i],train1_90))
for i in xrange(0,len(train1_180.index)) :
    testDeltaP180 = np.append(testDeltaP180, computeDelta(weight,test_180.iloc[i],train1_180))
for i in xrange(0,len(train1_360.index)) :
    testDeltaP360 = np.append(testDeltaP360, computeDelta(weight,test_360.iloc[i],train1_360))

# Actual deltaP values for test data.
testDeltaP = np.asarray(test_360[['Yi']])
testDeltaP = np.reshape(testDeltaP, -1)

# Combine all the test data
d = {'deltaP': testDeltaP,
     'deltaP90': testDeltaP90,
     'deltaP180': testDeltaP180,
     'deltaP360': testDeltaP360}
testData = pd.DataFrame(d)

# Predict price variation on the test data set.
result = model.predict(testData)
compare = { 'Actual': testDeltaP,
            'Predicted': result }
compareDF = pd.DataFrame(compare)

#Computation of MSE
MSE = 0.0
MSE = sm.mean_squared_error(testDeltaP, result)

print "The MSE is %f" % (MSE)
