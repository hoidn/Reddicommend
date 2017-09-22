import pyspark
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry
from pyspark.sql.types import *
from operator import add
from pyspark.mllib.linalg import Vectors, ArrayType
import numpy as np

def memo(f):
    """
    memoize, using object ids as cache keys
    """
    cache = {}
    def new_f(*args, **kwargs):
        ids = tuple(map(id, args)) + tuple(map(id, kwargs))
        if ids not in cache:
            cache[ids] = f(*args, **kwargs)
        return cache[ids]
    return new_f

sc = pyspark.context.SparkContext.getOrCreate()

def ndarr_to_coord_array(arr):
    entries = []
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] != 0:
                entries.append((i, j, arr[i][j]))
    return CoordinateMatrix(sc.parallelize(entries))

test_array = np.array([[1, 2, 3], [4, 0, 6], [0, 0, 8]], dtype = 'float')
test_array2 = np.array([[1, 2, 3, 0], [4, 0, 6, 1], [0, 0, 8, 2]], dtype = 'float')
test_array1d = np.array([[1, 1, 1, 1]])
test_array3 = np.array([[ 0.63203118,  0.30233108,  0.40677762,  0.58962667],
       [ 0.98905039,  0.9516414 ,  0.20273982,  0.20800506],
       [ 0.7751541 ,  0.94623161,  0.22601002,  0.40736821]])
test_coordmat = ndarr_to_coord_array(test_array)
test_coordmat2 = ndarr_to_coord_array(test_array2)
test_coordmat2_T = ndarr_to_coord_array(test_array2.T)

# TODO: check that zero entries are correctly filtered
def coordinateMatrixMultiply(leftmat, rightmat, zero_threshold = 0.):
    m = leftmat.entries.map(lambda entry: (entry.j, (entry.i, entry.value)))
    n = rightmat.entries.map(lambda entry: (entry.i, (entry.j, entry.value)))
    product_entries = m.join(n)\
    .map(lambda tup: ((tup[1][0][0], tup[1][1][0]), (tup[1][0][1] * tup[1][1][1])))\
    .filter(lambda tup: abs(tup[1]) >= zero_threshold)\
    .reduceByKey(add)\
    .filter(lambda tup: abs(tup[1]) >= zero_threshold)\
    .map(lambda record: MatrixEntry(record[0][0], record[0][1], record[1]))\
    .persist(pyspark.storagelevel.StorageLevel.MEMORY_AND_DISK_SER)
    
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(product_entries)

def eval_matrix_binop(ndarr1, ndarr2, op, **kwargs):
    cmat = op(ndarr_to_coord_array(ndarr1),
             ndarr_to_coord_array(ndarr2), **kwargs)
    ndmat = coordinate_matrix_to_ndarr(cmat)
    return cmat, ndmat

def test_multiply():
    assert np.all(coordinate_matrix_to_ndarr(coordinateMatrixMultiply(test_coordmat, test_coordmat)) ==\
        np.dot(test_array, test_array))
    assert np.all(coordinate_matrix_to_ndarr(coordinateMatrixMultiply(test_coordmat2, test_coordmat2_T)) ==\
        np.dot(test_array2, test_array2.T))

def coordinateMatrixAdd(leftmat, rightmat, scalar):
    """
    Return leftmat + scalar * rightmat
    """
    m = leftmat.entries.map(lambda entry: ((entry.i, entry.j), entry.value))
    n = rightmat.entries.map(lambda entry: ((entry.i, entry.j), scalar * entry.value))
    matsum = m.fullOuterJoin(n)\
    .map(lambda tup: MatrixEntry(tup[0][0], tup[0][1],
                                 reduce(add, filter(lambda elt: elt is not None, tup[1]))))
    
    #return matsum
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(matsum)

def test_add():
    m = coordinate_matrix_to_ndarr(
        coordinateMatrixAdd(ndarr_to_coord_array(test_array3), ndarr_to_coord_array(test_array2), -2))
    m2 = test_array3 - 2 * test_array2
    #return m, m2
    assert np.all(m == m2)

def coordinate_matrix_elementwise_vector_division(mat, vec):
    """
    mat : CoordinateMatrix
    
    mat_{ij} -> mat_{ij}/vec_{i}
    """
    m = mat.entries.map(lambda entry: (entry.i, (entry.j, entry.value)))
    v = vec.entries.map(lambda entry: (entry.i, (entry.j, entry.value)))
    matdiv = m.join(v).map(lambda tup: MatrixEntry(tup[0], tup[1][0][0], float(tup[1][0][1]) / tup[1][1][1]))
    
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(matdiv)

#from operator import multipl
def coordinate_matrix_elementwise_matrix_multiplication(mat1, mat2):
    """
    mat : CoordinateMatrix
    
    return matprod, where matprod_{ij} = mat1_{ij} * mat2_{ij}
    """
    m1 = mat1.entries.map(lambda entry: ((entry.i, entry.j), entry.value))
    m2 = mat2.entries.map(lambda entry: ((entry.i, entry.j), entry.value))
    
    matprod = m1.join(m2).map(lambda tup: MatrixEntry(tup[0][0], tup[0][1], tup[1][0] * tup[1][1]))
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(matprod)

def coordinate_matrix_sumj(mat):
    """
    mat : CoordinateMatrix
    """
    summed_entries = mat.entries.map(lambda entry: (entry.i, entry.value)).reduceByKey(add)\
                          .map(lambda tup: MatrixEntry(tup[0], 0, tup[1]))

    return pyspark.mllib.linalg.distributed.CoordinateMatrix(summed_entries)

def coordinate_matrix_row(mat, i):
    """
    mat : CoordinateMatrix
    
    return the specified row vector
    """
    filtered_entries = mat.entries.filter(lambda entry: entry.i == i).\
        map(lambda entry: MatrixEntry(0, entry.j, entry.value))
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(filtered_entries)

def coordinate_vector_matrix_norm(vec):
    """
    TODO: type checking, confusing name?
    """
    return np.sqrt(vec.entries.map(lambda entry: entry.value**2).sum())

def coordinate_matrix_vector_l2(vec1, vec2):
    """
    Given two vectors of the data type CoordinateMatrix, return L2 norm of vec1/|vec1| - vec2/|vec2|
    """
    norm1, norm2 = map(coordinate_vector_matrix_norm, [vec1, vec2])

    vec1normed = coordinateMatrixScalarMult(vec1, 1./norm1)
    vec2normed = coordinateMatrixScalarMult(vec2, 1./norm2)

    diff = coordinateMatrixAdd(vec1normed, vec2normed, -1.)
    return coordinate_vector_matrix_norm(diff)

def sort_row_indices_by_distance(mat, vec):
    """
    Return a list of the row indices of mat sorted by the ascending L2 distance between normalized row vectors and vec/|vec|
    """
    size = mat.numCols()
    row_vectors = mat.entries.map(lambda entry: (entry.i, [(entry.j, entry.value)]))\
        .reduceByKey(add).map(lambda tup: (tup[0], Vectors.sparse(size, tup[1])))
    # TODO replace all 1D CoordinateMatrix instances by local sparse vectors
    compare_vector = vec.entries.map(lambda entry: ('', [(entry.j, entry.value)])).reduceByKey(add)\
        .map(lambda tup: Vectors.sparse(size, tup[1])).collect()[0]
    # TODO check vector normalization
    return row_vectors\
        .map(lambda tup: (tup[1].squared_distance(compare_vector)/(tup[1].norm(2) * compare_vector.norm(2)), tup[0])).sortByKey()\
        .map(lambda tup: tup[1]).collect()


def coordinatematrix_get_row(mat, i):
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(
                    mat.entries.filter(lambda entry: entry.i == i)\
                    .map(lambda entry: MatrixEntry(0, entry.j, entry.value)))

@memo
def coordinatematrix_sort_rows(mat, elts_per_row = 10):
    """
    Return a dict mapping row indices to a list of column indices that sorts the row in reverse order. Only elts_per_row indices are included per row.
    """
    return dict(mat.entries.map(lambda x: (x.i, [(x.j, x.value)]))\
        .reduceByKey(add)\
        .sortByKey()\
        .map(lambda row: (row[0], map(lambda tup: tup[0], sorted(row[1], key = lambda tup: -tup[1]))[:elts_per_row])).persist(pyspark.storagelevel.StorageLevel.MEMORY_AND_DISK_SER)\
        .collect())


def transpose_coordinatematrix(mat):
    new_entries = mat.entries.map(lambda entry: (entry.j, entry.i, entry.value))
    return CoordinateMatrix(new_entries)

def coordinate_matrix_diagonal(mat):
    size = min(mat.numRows(), mat.numCols())
    new_entries = mat.entries.filter(
        lambda entry: entry.i == entry.j).map(
        lambda entry: (entry.i, entry.value)).collect()
    return Vectors.sparse(size, new_entries)

def test_coordinate_matrix_diagonal():
    mat = ndarr_to_coord_array(np.array([[1, 0], [0, .5]]))
    assert coordinate_matrix_diagonal(mat) == Vectors.sparse(2, {0: 1.0, 1: 0.5})

def test_transpose():
    arr = np.array([[1, 1], [0, 1]], dtype = 'float')
    T = coordinate_matrix_to_ndarr(transpose_coordinatematrix(ndarr_to_coord_array(arr)))
    assert np.all(T == arr.T)

def identity(mat):
    """
    return id matrix matching the shape of mat.
    """
    entries = [MatrixEntry(i, i, 1.) for i in xrange(min(mat.numRows(), mat.numCols()))]
    return CoordinateMatrix(sc.parallelize(entries))

def test_identity():
    arr = np.array([[1, 1], [0, 1]], dtype = 'float')
    d = coordinate_matrix_to_ndarr(diagonal(ndarr_to_coord_array(arr)))
    assert np.all(d == np.array([[ 1.,  0.], [ 0.,  1.]]))

def coordinatematrix_to_sparse_vector(mat):
    """
    mat : CoordinateMatrix
    
    Mat is assumed to have non-zero entries in only the 0th row index
    """
    size = mat.numCols()
    return Vectors.sparse(size, mat.entries.map(lambda entry: (entry.j, entry.value)).collect())

def ndarray_to_sparse_vector(arr):
    assert len(arr.shape) == 1
    size = len(arr)
    elts = [(i, value) for i, value in enumerate(arr) if value !=0]
    return Vectors.sparse(size, elts)

def sparse_vector_to_ndarray(vec):
    arr = np.zeros(vec.size)
    for i, val in zip(vec.indices, vec.values):
        arr[i] = val
    return arr
    
@memo
def coordinate_matrix_to_ndarr(mat):
    size = mat.entries.count()
    elts = mat.entries.take(size)
    arr = np.zeros((mat.numRows(), mat.numCols()))
    for elt in elts:
        arr[elt.i][elt.j] = elt.value
    return arr

def coordinate_matrix_divide_by_sparse_vector(mat, vect):
    """
    mat : CoordinateMatrix
    
    vec : Vectors.sparse
    
    mat_{ij} -> mat_{ij}/vec_{i}
    """
    m = mat.entries.map(lambda entry: (entry.i, (entry.j, entry.value)))
    v = sc.parallelize(list(zip(vect.indices, vect.values)))
    matdiv = m.join(v).map(lambda tup: MatrixEntry(tup[0], tup[1][0][0], tup[1][0][1]/tup[1][1]))
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(matdiv)

def test_coordinate_matrix_divide_by_sparse_vector():
    mat = ndarr_to_coord_array(test_array)
    vec = ndarray_to_sparse_vector(np.array([1, 2, 3]))
    res = coordinate_matrix_to_ndarr(coordinate_matrix_divide_by_sparse_vector(mat, vec))
    assert np.all(np.isclose(res, np.array([[ 1.        ,  2.        ,  3.        ],
           [ 2.        ,  0.        ,  3.        ],
           [ 0.        ,  0.        ,  2.66666667]])))

def coordinatematrix_multiply_vector_elementwise(mat, vec):
    """
    mat : CoordinateMatrix
    vec : CoordinateMatrix
    """
    mat_entries = mat.entries.map(lambda entry: (entry.i, (entry.j, entry.value)))
    vec_entries = vec.entries.map(lambda entry: (entry.i, entry.value))

    prod = vec_entries.join(mat_entries).map(lambda tup: MatrixEntry(tup[0], tup[1][1][0], tup[1][0] * tup[1][1][1]))
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(prod)

def coordinateMatrixElementwise(mat, op):
    """
    elt -> op(elt) for each nonzero element elt of the matrix mat
    """
    new_entries = mat.entries.map(lambda entry: MatrixEntry(entry.i, entry.j, op(entry.value)))
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(new_entries)

def sparse_vector_elementwise(vec, op):
    new_entries = [(i, op(val)) for i, val in zip(vec.indices, vec.values)]
    return Vectors.sparse(vec.size, new_entries)

def test_sparse_vector_elementwise():
    np.all(np.isclose(sparse_vector_to_ndarray(sparse_vector_elementwise(ndarray_to_sparse_vector(np.array([1, 2, 3])),
        lambda elt: np.sqrt(elt))), np.array([ 1.        ,  1.41421356,  1.73205081])))

def coordinateMatrixElementwiseMultiplication(mat, scalar):
    """
    return scalar * mat
    """
    new_entries = mat.entries.map(lambda entry: MatrixEntry(entry.i, entry.j, scalar * entry.value))
    return pyspark.mllib.linalg.distributed.CoordinateMatrix(new_entries)


