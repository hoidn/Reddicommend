import pyspark
import numpy as np

import linalg as la
reload(la)
epsilon = 1e-9

def similarity_matrix(activity):
    sim = la.coordinateMatrixMultiply(
        activity,
        la.transpose_coordinatematrix(activity))
    print 'sim', la.coordinate_matrix_to_ndarr(sim)
    diag =  la.coordinate_matrix_diagonal(sim)
    print 'diag', diag
    norms = la.sparse_vector_elementwise(diag, lambda num: np.sqrt(num))
    print la.sparse_vector_to_ndarray(norms)

    sim_normed = la.coordinate_matrix_divide_by_sparse_vector(
            sim, norms)

    return la.transpose_coordinatematrix(
        la.coordinate_matrix_divide_by_sparse_vector(
            la.transpose_coordinatematrix(sim_normed),
            norms))

def test_similarity_matrix():
    input = np.array([[ 0.47364486,  0.61937755,  0.00844025,  0.65533834,  0.77267817,
         0.37602224],
       [ 0.13680721,  0.6778712 ,  0.47219011,  0.10933562,  0.66098062,
         0.93689595],
       [ 0.85607774,  0.56905013,  0.92456913,  0.15854798,  0.84589546,
         0.22489419],
       [ 0.64035207,  0.90656327,  0.42716786,  0.64355359,  0.4160582 ,
         0.30400444],
       [ 0.78897855,  0.06087785,  0.42894637,  0.99971801,  0.63067103,
         0.19627554]])
    expected = np.array([[ 1.        ,  0.75002058,  0.73386657,  0.89499209,  0.81681218],
           [ 0.75002058,  1.        ,  0.73751518,  0.74443305,  0.49754955],
           [ 0.73386657,  0.73751518,  1.        ,  0.83284528,  0.74795564],
           [ 0.89499209,  0.74443305,  0.83284528,  1.        ,  0.78808545],
           [ 0.81681218,  0.49754955,  0.74795564,  0.78808545,  1.        ]])
    assert np.all(np.isclose(la.coordinate_matrix_to_ndarr(similarity_matrix(la.ndarr_to_coord_array(input))), expected))
