from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np

cdef extern from "generateTemporalCrossCorrelogram.c":
    void  binnedISI(double *spikeTs_1, double *spikeTs_2, long *ISICounts, int spikeTsLength_1, int spikeTsLength_2)
    
def getBinnedISI(np.ndarray spikeTimestamps_1, int spikeTimestampsLength_1, np.ndarray spikeTimestamps_2, int spikeTimestampsLength_2):
    ISI = np.array([0]*2001)
    cdef int spikeTsLength_1 = spikeTimestampsLength_1
    cdef int spikeTsLength_2 = spikeTimestampsLength_2
    cdef double[:] spikeTs_1 = spikeTimestamps_1.reshape((spikeTimestamps_1.size,))
    cdef double[:] spikeTs_2 = spikeTimestamps_2.reshape((spikeTimestamps_2.size,))
    cdef long[:] ISICounts = ISI.reshape((ISI.size,))
    binnedISI(&spikeTs_1[0], &spikeTs_2[0], &ISICounts[0], spikeTsLength_1, spikeTsLength_2)
    return ISICounts
