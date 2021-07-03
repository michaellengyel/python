import numpy as np

def main():

    # Reference:
    # https://numpy.org/doc/stable/user/quickstart.html

    ### Numpy Array Basics ###
    print("Numpy Array Basics:")

    # Creating a numpy array
    a = np.array([1, 2, 3])
    print(a)

    # Creating a multidimensional numpy array
    b = np.array([[1, 2, 3], [3, 2, 1]])
    print(b)

    # Creating a multidimensional numpy array with a specific data type
    c = np.array([[1, 2, 3], [3, 2, 1]], dtype=float)
    print(c)

    # Creating a numpy array from a set
    d = np.array([(1, 2, 3)])
    print(d)

    # Create a numpy array of a certain shape initialized with zeros
    e = np.zeros((3, 4))
    print(e)

    # Create a numpy array of a certain shape initialized with ones
    f = np.ones((5, 4), dtype=np.int16)
    print(f)

    # Create a numpy array of a certain shape initialized with nothing
    g = np.empty((5, 2))
    print(g)

    eye = np.eye(4, 4)
    print(eye)

    # Create a numpy array with (from, to, increment)
    h = np.arange(2., 31., 3.32)
    print(h)

    # For floating point numbers, use linspace (from, to, into number of parts)
    i = np.linspace(2., 31.3, 4)
    print(i)

    # To reshape a numpy array
    j = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
    print(j)

    # To get the type and shape of an array
    print("Get the type and shape of an array")
    print(j.dtype)
    print(j.shape)

    ### Numpy array operations ###
    print("Numpy array operations:")

    k = np.array([[1, 2], [3, 4]])
    l = np.array([[1, 2], [3, 4]])

    # Element wise addition
    print(k + l)

    # Element wise multiplication
    print(k * l)

    # Matrix multiplication
    print(k.dot(l))

    # Sum of each column
    print("Sum of each col: ", k.sum(axis=0))

    # Sum of each row
    print("Sum of each row: ", k.sum(axis=1))

    # Min of each row
    print("Min of each row: ", k.min(axis=0))

    # Cumulative sum
    print("Cumulative sum of each row")
    print(k.cumsum(axis=0))
    print("Cumulative sum of each column")
    print(k.cumsum(axis=1))

    ### Numpy universal functions ###
    print("Numpy universal functions:")

    """
    all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov,
    cross, cumprod, cumsum, diff, dot, floor, inner, invert, lexsort, max, maximum, mean, median, min, minimum,
    nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where
    """

    m = [[1, 2], [3, 4]]
    n = [[3, 3], [3, 3]]
    # Numpy element wise exponential

    print("Exponential:",)
    print(np.exp(m))
    print("Sinus:",)
    print(np.sin(m))
    print("Square Root: ")
    print(np.sqrt(m))
    print("Addition: ")
    print(np.add(m, n))

    ### Numpy Indexing, Slicing and Iterating ###
    print("Numpy Indexing, Slicing and Iterating:")

    """
    ndarray.shape, reshape, resize, ravel
    """

    o = np.arange(10)**3
    print(o)
    print(o[3])
    print(o[2:5])
    print(o[5:])

    # "From position 2 to position 5, set every element to 999
    o[2:5] = 999
    print(o)

    # "From start to position 7, set every 2nd element to 0
    o[:7:2] = 0
    print(o)

    o = o[::-1]
    print(o)

    print("Cropping for 2D array:")
    p = np.arange(20).reshape(4, 5)
    print(p)
    print(p[:, 2])
    print(p[1, :])
    print(p[1:3, 1:3])
    print(p[-1])

    print("Cropping for 3D array:")
    # The ... notation
    q = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    print(q)
    print(q[:, :, 1])
    # Equivalently:
    print(q[..., 1])

    print("Iterating through all elements of an n-dim array:")
    for element in q.flat:
        print(element)

    "Numpy shape change:"
    print("Numpy shape change:")
    r = np.arange(10).reshape(2, 5)
    print(r)
    print(r.ravel())
    print(r.reshape(5, 2))
    print(r.T)
    print(r.T.shape)

    print("Numpy shape change, resize vs reshape:")
    # Use reshape when the total size of the data does not change
    # Use resize when the original size of the array is changes.

    t = np.array([[3., 7., 3., 4.], [1., 4., 2., 2.], [7., 2., 4., 9.]])
    print(t)

    # resize causes 0s padding
    t.resize((7, 7))
    print(t)

    # Stacking together different arrays
    print("Stacking together different arrays:")

    u = np.array([[1, 2], [3, 4]])
    v = np.array([[1, 2], [3, 4]])
    print(np.vstack((u, v)))
    print(np.hstack((u, v)))

    # Splitting arrays
    print("Splitting arrays:")
    w = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    print(w)

    y1, y2, y3 = np.hsplit(w, 3)
    print(y1)
    print(y2)
    print(y3)

    z1, z2 = np.vsplit(w, 2)
    print(z1)
    print(z2)


if __name__ == '__main__':
    main()
