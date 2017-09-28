import math as mt
import csv
from  scipy.sparse.linalg import svds as sparsesvd
#from sparsesvd import sparsesvd
import numpy as np
from scipy.sparse import csr_matrix
import pdb
import linalg

#constants defining the dimensions of our User Rating Matrix (URM)
MAX_PID = 37143
MAX_UID = 15375

def readUrm():
    urm = np.random.random((120, 100))
    #np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)
#    with open('/PathToTrainFile.csv', 'rb') as trainFile:
#        urmReader = csv.reader(trainFile, delimiter=',')
#        for row in urmReader:
#            urm[int(row[0]), int(row[1])] = float(row[2])

    return csr_matrix(urm, dtype=np.float32)

def readUsersTest():
    uTest = dict()
    with open("./testSample.csv", 'rb') as testFile:
        testReader = csv.reader(testFile, delimiter=',')
        for row in testReader:
            uTest[int(row[0])] = list()

    return uTest


def getMoviesSeen():
    moviesSeen = dict()
    with open("./trainSample.csv", 'rb') as trainFile:
        urmReader = csv.reader(trainFile, delimiter=',')
        for row in urmReader:
            try:
                moviesSeen[int(row[0])].append(int(row[1]))
            except:
                moviesSeen[int(row[0])] = list()
                moviesSeen[int(row[0])].append(int(row[1]))

    return moviesSeen



@linalg.memo
def computeSVD(urm, K):
    U, s, Vt = sparsesvd(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i, i] = mt.sqrt(s[i])

    return U, S, Vt
    S = csr_matrix(S)

    return csr_matrix(U), S, csr_matrix(Vt)    


from scipy.sparse.linalg import * #used for matrix multiplication

def computeEstimatedRatings(urm, U, S, Vt, uTest, moviesSeen, K, test):
    pdb.set_trace()
    rightTerm = S*Vt 

    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm

        #we convert the vector to dense format in order to get the indices of the movies with the best estimated ratings 
        estimatedRatings[userTest, :] = prod.todense()
        recom = (-estimatedRatings[userTest, :]).argsort()[:250]
        for r in recom:
            if r not in moviesSeen[userTest]:
                uTest[userTest].append(r)

                if len(uTest[userTest]) == 5:
                    break

    return uTest

def main():
    K = 90
    urm = readUrm()
    U, S, Vt = computeSVD(urm, K)
    uTest = readUsersTest()
    moviesSeen = getMoviesSeen()
    uTest = computeEstimatedRatings(urm, U, S, Vt, uTest, moviesSeen, K, True)

def s_mat(singular_values):
    s = np.zeros((len(singular_values), len(singular_values)))
    for i, val in enumerate(singular_values):
        s[i][i] = val
    return csr_matrix(s)

def svd_correlation_matrix(arr, k = 100, epsilon=1e-9, normalize = True):
    U, S, Vt = computeSVD(csr_matrix(arr), k)
    #lowrank_subs = np.dot(S, U.T)
    lowrank_subs = np.dot(U, S)
    sim = np.dot(lowrank_subs, lowrank_subs.T)
    #diag = sim.diagonal()
    diag = np.diagonal(sim)
    if normalize:
        norms = np.array([np.sqrt(np.abs(diag))])
        return np.diagonal(S), sim / norms / norms.T
    else:
        return np.diagonal(S), sim

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
