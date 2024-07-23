""" The :py:mod:`pycqed.src.util` module defines a series of utility functions that are used internally and can be used by the user.
"""
import qutip as qt
import numpy as np
import sympy as sy
import scipy as sc
import itertools as itt
import pickle

def mdot(*args):
    """ Compute a sequence of dot or matrix products.
    
    :param \*args: A sequence of numpy matrices to multiply in the order specified.
    :type \*args: np.ndarray, np.ndarray ...
    
    :return: The result of the dot or matrix product.
    :rtype: numpy.ndarray
    """
    res = args[-1]
    ret = 0.0
    for i in range(len(args)-1):
        ret = np.dot(args[-(i+2)],res)
        res = ret
    return ret

def hdot(m1, m2):
    """ Compute the Hadamard product of two matrices.
    
    :param m1: The first matrix.
    :type m1: np.ndarray, np.matrix, list
    
    :param m2: The second matrix.
    :type m2: np.ndarray, np.matrix, list
    
    :return: The Hadamard product of two matrices.
    :rtype: np.matrix
    """
    return np.matrix(np.array(m1) * np.array(m2))

def sanArray(a, tol=1e-10):
    """ Sanitizes an array `a` by setting values less than `tol` to zero, including the imaginary component.
    
    :param a: The input array.
    :type a: np.ndarray, np.matrix, list
    
    :param tol: The cutoff below which values are set to zero.
    :type tol: float
    """
    arr = np.array(a).copy()
    arr.real[np.abs(arr.real) < tol] = 0.0
    try:
        arr.imag[np.abs(arr.imag) < tol] = 0.0
    except ValueError:
        pass
    if type(a) is list:
        return list(arr)
    else:
        return arr

def sanFloat(f, tol=1e-10):
    """ Sanitizes a float `f` by setting it to zero if less than `tol`, including the imaginary component.
    
    :param f: The input float.
    :type f: float
    
    :param tol: The cutoff below which values are set to zero.
    :type tol: float
    """
    temp = 0.0
    if np.abs(f.real) > tol:
        temp += f.real
    if np.abs(f.imag) > tol:
        temp += f.imag
    return temp

def diagSparseH(M, eigvalues=5, get_vectors=False, sparsesolveropts={"sigma":None, "mode":"normal", "maxiter":None, "tol":0}):
    """ Sparse Hermitian matrix diagonalizer. Wraps `scipy.sparse.linalg.eigsh`.
    
    :param M: The Hermitian matrix to diagonalize.
    :type M: qutip.qobj.Qobj
    
    :param eigvalues: The number of lowest eigenvalues (and vectors if `get_vectors` is `True`) to compute.
    :type eigvalues: int, optional
    
    :param get_vectors: Whether to get the associated eigenvectors.
    :type get_vectors: bool, optional
    
    :param sparsesolveropts: A dictionary of keyword arguments to pass to the sparse solver, see the documentation of `scipy.sparse.linalg.eigsh` for details.
    :type sparsesolveropts: dict, optional
    
    :raises Exception: If `M` is not a qutip.qobj.Qobj instance or is not Hermitian.
    
    :return: A sorted list of eigenvalues or a tuple of eigenvalues and normalised eigenvectors (as qutip.qobj.Qobj types) if `get_vectors` is `True`.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    if type(M) != qt.qobj.Qobj:
        raise Exception("not a qutip Qobj instance.")
    if not M.isherm:
        raise Exception("matrix object is not Hermitian.")
    
    # Diagonalize with sparse matrix
    ret = sc.sparse.linalg.eigsh(M.to("CSR").data.as_scipy(), k=eigvalues, return_eigenvectors=get_vectors, **sparsesolveropts)
    
    # Sort the results
    if get_vectors:
        E, V = ret
        
        # Sort the eigenvalues and use new indices to sort vectors
        _zipped = list(zip(E, range(eigvalues)))
        _zipped.sort()
        E, perm = list(zip(*_zipped))
        
        # Convert the vectors to Qobj while sorting and set their dimensions based on that of the original operator
        Vt = np.empty(len(perm), dtype=qt.qobj.Qobj)
        for i, k in enumerate(perm):
            Vt[i] = qt.Qobj(V[:, k], dims=[M.dims[0], [1] * len(M.dims[0])])
        #V = np.array([qt.Qobj(V[:, k], dims=[M.dims[0], [1] * len(M.dims[0])]) for k in perm], dtype=qt.qobj.Qobj)
        
        # Get the norm of the eigenvectors
        norms = np.array([ket.norm() for ket in Vt])
        
        return E, Vt/norms
    else:
        ret.sort()
        return ret

def diagDenseH(M, eigvalues=5, get_vectors=False, sparsesolveropts=None):
    """ Dense Hermitian matrix diagonalizer. Wraps `scipy.linalg.eigh`.
    
    :param M: The Hermitian matrix to diagonalize.
    :type M: qutip.qobj.Qobj
    
    :param eigvalues: The number of lowest eigenvalues (and vectors if `get_vectors` is `True`) to compute.
    :type eigvalues: int, optional
    
    :param get_vectors: Whether to get the associated eigenvectors.
    :type get_vectors: bool, optional
    
    :param sparsesolveropts: A dictionary of keyword arguments to pass to the sparse solver. These are not used here but are included to reduce if statements where the diagonalizer functions are used.
    :type sparsesolveropts: dict, optional
    
    :raises Exception: If `M` is not a qutip.qobj.Qobj instance or is not Hermitian.
    
    :return: A sorted list of eigenvalues or a tuple of eigenvalues and normalised eigenvectors (as qutip.qobj.Qobj types) if `get_vectors` is `True`.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    if type(M) != qt.qobj.Qobj:
        raise Exception("not a qutip Qobj instance.")
    #if not M.isherm:
    #    raise Exception("matrix object is not Hermitian.")
    
    # Diagonalize with dense matrix.
    ret = sc.linalg.eigh(M.data.to_array(), eigvals_only=(not get_vectors), subset_by_index=(0, eigvalues-1))
    
    # Sort the results
    if get_vectors:
        E, V = ret
        
        # Sort the eigenvalues and use new indices to sort vectors
        _zipped = list(zip(E, range(eigvalues)))
        _zipped.sort()
        E, perm = list(zip(*_zipped))
        
        # Convert the vectors to Qobj while sorting and set their dimensions based on that of the original operator
        Vt = np.empty(len(perm), dtype=qt.qobj.Qobj)
        for i, k in enumerate(perm):
            Vt[i] = qt.Qobj(V[:, k], dims=[M.dims[0], [1] * len(M.dims[0])])
        #V = np.array([qt.Qobj(V[:, k], dims=[M.dims[0], [1] * len(M.dims[0])]) for k in perm], dtype=qt.qobj.Qobj)
        
        # Get the norm of the eigenvectors
        norms = np.array([ket.norm() for ket in Vt])
        
        return E, Vt/norms
    else:
        ret.sort()
        return ret

def getACStarkShift(Erwa):
    """ Returns the circuit AC stark shift as a function of the average photon number in a linear resonator.
    
    :param Erwa: The full eigenvalue array returned by :func:`pycqed.src.HamilSpec.getResonatorResponse`.
    :type Erwa: numpy.ndarray
    
    :return: The photon number list and the AC Stark shifted circuit energy gaps.
    :rtype: (numpy.ndarray, numpy.ndarray)
    
    The format of the returned array is: `ret[q,n]` is the gap between the state `q` and the ground state, at `n` photons.
    """
    # Get number of levels
    levels = Erwa.shape[0]
    
    # Get max number of photons
    photons = Erwa.shape[1] - Erwa.shape[0]
    
    # Get all the gaps with ground state
    transitions = np.array([[Erwa[q+1,n+q+1]-Erwa[0,n] for n in range(photons)] for q in range(levels-1)])
    
    # Rearrange
    acstark = [[transitions[q,n] for n in range(photons)] for q in range(levels-1)]
    return np.array(list(range(photons))),np.array(acstark)

def getCircuitLambShift(Erwa):
    """ Returns the circuit Lamb shift due to coupling to a linear resonator.
    
    :param Erwa: The full eigenvalue array returned by :func:`pycqed.src.HamilSpec.getResonatorResponse`.
    :type Erwa: numpy.ndarray

    :return: The Lamb shifted (dressed) circuit energy gaps.
    :rtype: numpy.ndarray
    
    The format of the returned array is: `ret[q]` is the gap between state `q` and the ground state, at 0 photons.
    """
    
    # Get number of levels
    levels = Erwa.shape[0]
    return np.array([Erwa[i,i] for i in range(levels)])

def getResonatorShift(Erwa):
    """ Returns the linear resonator shift as a function of the average photon number.
    
    :param Erwa: The full eigenvalue array returned by :func:`pycqed.src.HamilSpec.getResonatorResponse`.
    :type Erwa: numpy.ndarray
    
    :return: The Lamb shifted (dressed) circuit energy gaps.
    :rtype: numpy.ndarray
    
    The format of the returned array is: `ret[q,n]` is the resonant frequency at circuit state `q` and at `n` photons.
    """
    # Get number of levels
    levels = Erwa.shape[0]
    
    # Get max number of photons
    photons = Erwa.shape[1] - Erwa.shape[0]
    
    # Get all the gaps with ground state
    transitions = np.array([[Erwa[q,n+q+1]-Erwa[q,n+q] for n in range(photons)] for q in range(levels-1)])
    
    # Rearrange
    res = [[transitions[q,n] for n in range(photons)] for q in range(levels-1)]
    return np.array(res)

def isStoquastic(H, order=[2,3]):
    """ Checks if the given Hamiltonian is Stoquastic. Non-Stoquastic Hamiltonians are understood to lead to sign-problems in Quantum Monte Carlo methods :cite:`Gupta2019b`. We use the permutation matrix representation of the Hamiltonian and the Off-Diagonal Expansion method :cite:`Gupta2019a` to find the configurations for which a negative weight exists. There are infinite possible configurations to infinite order. It should suffice to check the configurations of low order, however the order values to check can be specified.
    
    __IMPLEMENTATION NOTE:__ Currently the implementation is inefficient and cannot make use of sparse matrices.
    
    :param H: The Hamiltonian to verify stoquasticity.
    :type H: qutip.qobj.Qobj
    
    :param order: A list of order values for which each configuration should be checked.
    :type order: list, optional
    
    :return: True if the specified Hamiltonian is stoquastic.
    :rtype: bool
    """
    
    # Get the Hamiltonian dimension
    H = H.data.to_array()
    M = H.shape[0]
    
    # Get permutation matrices
    qj = _get_permutation_matrices(M)

    # Decompose Hamiltonian
    Qj = [hdot(H,qv) for qv in qj]

    # Get the diagonal and new permutation matrices
    Dj, Pj = _get_decomposition(Qj,qj)

    # Get computational basis states
    z_basis = _get_basis_states(Dj[0])

    # Get all cyclic sequences for each order specified
    for q in order:
        seqs = _get_unique_sequences(M,q)

        # Check all coefficients of cyclic configurations
        for z in range(M):
            for seq in seqs:
                
                # If we find a single negative coefficient then we could get sign problems
                if _get_coefficient(M,z,seq,z_basis,Dj).real < 0.0:
                    return False
    return True

def createSubspaceOperators(m00, m01, m10, m11):
    """
    """
    Op = None
    ret = []
    for i in range(len(m00)):
        Op = np.asmatrix(np.eye(2), dtype=np.complex64)
        Op[0, 0] = m00[i]
        Op[0, 1] = m01[i]
        Op[1, 0] = m10[i]
        Op[1, 1] = m11[i]
        ret.append(Op)
    return ret

def pauliCoefficients(E, V, basis_op):
    """ Performs Hamiltonian reduction to a two-dimensional Hilbert space useful for describing qubits as spins. This function returns the Pauli coefficients :math:`h_\mathrm{x,y,z}` of the corresponding spin operators :math:`\hat{\sigma}_\mathrm{x,y,z}`. We use the 'local basis' approach developed by G. Consani :cite:`Consani2019` which is superior to traditional qubit circuit Hamiltonian reduction methods.
    
    :param E: A two-dimensional list of eigenvalues, the first index corresponding to the state number, and the second corresponding to the value of a swept variable, usually an external charge or flux depending on the qubit type.
    :type E: numpy.ndarray
    
    :param V: The eigenvectors corresponding to the supplied eigenvalues. The eigenvectors themselves should be `Qobj` instances.
    :type V: numpy.ndarray
    
    :param basis_op: A list or a single operator that defines the computational basis. Usually the persistent current for a flux qubit or the charge number for a charge qubit.
    :type basis_op: qutip.qobj.Qobj, numpy.ndarray
    
    :raises Exception: If `E` and `V` are not of type `list` or `numpy.ndarray`.
    
    :return: The list of :math:`h_\mathrm{x,y,z}` values for each value of the swept variable.
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    
    def get_subspace_operator(V0, V1, Oq, i):
        Op = np.asmatrix(np.eye(2), dtype=np.complex64)
        Op[0, 0] = Oq.matrix_element(V0, V0)
        Op[0, 1] = Oq.matrix_element(V0, V1)
        Op[1, 0] = Oq.matrix_element(V1, V0)
        Op[1, 1] = Oq.matrix_element(V1, V1)
        return Op
    
    def ret_subspace_operator(V0, V1, Oq, i):
        return Oq[i]
    
    op_func = None
    if type(basis_op) == qt.qobj.Qobj and type(E) in [list, np.ndarray] and type(V) in [list, np.ndarray]:
        #Oq = [basis_op]*len(E[0]) # This should be too memory intensive as its the same object reference repeated.
        op_func = get_subspace_operator
    elif type(basis_op) in [list, np.ndarray] and type(E) in [list, np.ndarray] and type(V) in [list, np.ndarray]:
        #Oq = basis_op
        op_func = ret_subspace_operator
    else:
        raise Exception("incompatible input types for E (%s), V (%s) and basis_op (%s)." % (type(E), type(V), type(basis_op)))
    
    # Get Paulis
    ox = np.asmatrix(qt.sigmax().data.to_array(), dtype=np.complex64)
    oy = np.asmatrix(qt.sigmay().data.to_array(), dtype=np.complex64)
    oz = np.asmatrix(qt.sigmaz().data.to_array(), dtype=np.complex64)
    
    # Get ground and first excited states
    E0 = E[0, :]
    E1 = E[1, :]
    V0 = V[0, :]
    V1 = V[1, :]
    
    # Get the Pauli prefactors
    hx = []
    hy = []
    hz = []
    Hq = None
    Hqp = None
    El = None
    Vl = None
    U = None
    Udag = None
    Op = None
    for i in range(len(E0)):
        
        # Create computational basis subspace operator
        Op = op_func(V0[i], V1[i], basis_op, i)
        #Op = np.asmatrix(np.eye(2), dtype=np.complex64)
        #Op[0,0] = (V0[i].dag()*Oq[i]*V0[i])[0][0][0]
        #Op[0,1] = (V0[i].dag()*Oq[i]*V1[i])[0][0][0]
        #Op[1,0] = (V1[i].dag()*Oq[i]*V0[i])[0][0][0]
        #Op[1,1] = (V1[i].dag()*Oq[i]*V1[i])[0][0][0]
        
        # Get its eigenvectors as a matrix
        El, Vl = np.linalg.eigh(Op)
        
        # Apply the gauge transformation to the eigenvectors
        Vl = np.array([np.abs(Vl[0, k])/Vl[0, k] * Vl[:, k] for k in (0, 1)])
        
        # Create the unitary
        U = np.asmatrix(Vl.T, dtype=np.complex64)
        Udag = U.conjugate().T
        
        # Create Hq prime
        Hqp = np.asmatrix(np.diag([E0[i], E1[i]]), dtype=np.complex64)
        
        # Apply the transformation
        Hq = mdot(Udag, Hqp, U)
        
        # Do the projection
        hx.append(np.real(0.5*np.dot(Hq, ox).trace()[0, 0]))
        hy.append(np.real(0.5*np.dot(Hq, oy).trace()[0, 0]))
        hz.append(np.real(0.5*np.dot(Hq, oz).trace()[0, 0]))
    return np.array(hx), np.array(hy), np.array(hz)

def getEigenValuesAndVectors(sweep):
    """ Convenience function to separate the eigenvalues from the eigenvectors of a parameter sweep returned by :func:`pycqed.src.HamilSpec.getSweep`. When the diagonaliser is configured to return both eigenvalues and vectors, the dtype of the `numpy.ndarray` will be `object`, which results in conversion issues for example in `matplotlib`. This function performs the conversion of the `numpy.ndarray` dtype.
    
    :param sweep: The sweep structure returned by :func:`pycqed.src.HamilSpec.getSweep`.
    :type sweep: numpy.ndarray
    
    :return: The eigenvalues as `numpy.float64` types and eigenvectors as `object` types contained in `sweep`.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    return (sweep[:,0].astype(np.float64), sweep[:,1])

def getExpectationValues(sweep):
    """
    """
    pass

def pickleWrite(obj, filename):
    """ Write pickled data to a binary file.
    
    :param obj: A serialisable python object.
    :type obj: object
    
    :param filename: The path and filename to write to.
    :type filename: str
    
    :return: The filename that was written to.
    :rtype: str
    """
    # Open file for writing in binary
    fd = open(filename,"wb")
    
    # Pickle the object
    pickle.dump(obj,fd)
    
    # Write to file
    fd.close()
    
    return filename

## Read pickled data from binary file
#
#  @param filename The filename to use.
#  @return A python object.
#
def pickleRead(filename):
    """ Read pickled data from a binary file.
    
    :param filename: The path and filename to read from.
    :type filename: str
    
    :return: The unpickled data contained in the specified file.
    :rtype: object
    """
    # Read file in binary
    fd = open(filename,"rb")
    
    # Unpickle
    obj = pickle.load(fd)
    fd.close()
    
    return obj

# Generate nearest neighbours of HW 1
def getNeighbours(bitstring, HW=1):
    """ Generates the nearest neighbours of the given bit string according to the given Hamming weight.
    
    :param bitstring: The bitstring consisting of 0 and 1 characters.
    :type bitstring: str
    
    :param HW: The desired Hamming weight, defaults to 1.
    :type HW: int, optional
    
    :return: A list of the nearest neighbours of Hamming weight `HW` to `bitstring`.
    :rtype: list
    """
    neighbours = []
    bits = len(bitstring)
    if HW == 1:
        for i in range(bits):
            neighbours.append(
                "".join([bitstring[j] if i != j else str(int(not int(bitstring[j]))) for j in range(bits)])
            )
    return neighbours

# Generate state from bit string
def stateFromBitstring(bitstring):
    """ Generates the state vector associated with a bit string.
    
    :param bitstring: The bitstring consisting of 0 and 1 characters.
    :type bitstring: str
    
    :return: The eigenvector associated with `bitstring`.
    :rtype: Qobj
    """
    bits = len(bitstring)
    s = [0]*bits
    for i, b in enumerate(bitstring):
        s[i] = qt.basis(2, 1) if b == '0' else qt.basis(2, 0)
    return qt.tensor(*s)

#
# Internal
#
def _get_permutation_matrices(M):
    a = np.arange(M)
    perm_mat = []
    for j in range(M):
        mat = np.matrix([[0.0]*M]*M)
        b = np.roll(a,j)
        for i in range(M):
            mat[a[i],b[i]] = 1.0
        perm_mat.append(mat)
    return perm_mat

def _get_decomposition(perm_mat,qj):
    diag_mat = []
    for i,perm in enumerate(perm_mat):
        qjinv = np.linalg.inv(qj[i]) # THIS IS INEFFICIENT, SHOULDN'T HAVE TO DO THIS
        diag_mat.append(sanArray(mdot(perm,qjinv)))
    return diag_mat, qj

def _get_basis_states(D):
    E,V = np.linalg.eig(D)
    return [x.T for x in V]

def _get_k_ij_map(M):
    a = np.array(range(1,M))

    k = list(-a[::-1])
    k.extend(list(a))

    ij = list(a)
    ij.extend(list(a))

    k_ij_map = {}
    for i in range(len(k)):
        k_ij_map[k[i]] = ij[i]
    return k_ij_map

def _get_unique_sequences(M,q):
    k_ij = _get_k_ij_map(M)
    
    a = np.array(range(1,M))
    k = list(-a[::-1])
    k.extend(list(a))
    
    # Do the Cartesian product
    perms = [x for x in itt.product(k,repeat=(q))]
    
    # Find sequences that add up to zero
    seq = []
    for perm in perms:
        if sum(perm) == 0:
            ijs = [k_ij[i] for i in perm]
            if ijs not in seq:
                seq.append(ijs)
    return seq

def _get_state_indices(M,z0,seq):
    zacc = z0
    state_indices = []
    for ij in seq:
        zacc += ij
        state_indices.append((zacc)%(M))
    return state_indices

def _get_coefficient(M,z0,seq,z,Dj):
    
    # Get state lookup indices
    z_indices = _get_state_indices(M,z0,seq)
    
    # Go through the ij indices
    sub_coefs = []
    for c,ij in enumerate(seq):
        zj = z[z_indices[c]]
        sub_coefs.append(mdot(zj.transpose().conjugate(),Dj[ij],zj)[0,0])
    return np.prod(-np.array(sub_coefs))
