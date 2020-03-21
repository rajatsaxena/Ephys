# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 21:13:53 2019

@author: Rajat
"""

import warnings
warnings.filterwarnings('ignore')
from neurodsp import timefrequency as tf
import scipy.integrate as integrate
from sklearn import linear_model
import nitime.algorithms as tsa
import matplotlib.pyplot as plt
from scipy.special import betainc
import scipy.signal as scsig
from neurodsp import spectral
from neurodsp import filt
from neurodsp import burst
import neuralynxio as nlxio
from pacpy.pac import *
import bisect, math
import numpy as np

def zscore(x):
    return (x - x.mean()) / x.std()

# function to load the csc datasets
def load_cscdata(csc_name, maxlength=None):
    data = nlxio.load_ncs(csc_name)
    sig_data = data['data']
    fs = float(data['sampling_rate'])
    times = data['time']
    if maxlength is not None:
        sig_data = sig_data[:maxlength]
        times = times[:maxlength]
    times = times/1e6
    times = times - times[0]
    times = times
    return sig_data, times, fs
 
# function to get band pass filtered signal
# passtype = highpass, lowpass, bandpass & bandstop
def get_bandpass_filter_signal(data, fs, f_range,passtype='bandpass'):
    sig_filt = filt.filter_signal(data, fs, passtype, f_range, remove_edges=False)
    return sig_filt

# function to get instantaneous freq, phase and amplitude    
def get_inst_data(data, fs, f_range):
    i_phase = tf.phase_by_time(data, fs, f_range)
    i_amp = tf.amp_by_time(data, fs, f_range)
    i_freq = tf.freq_by_time(data, fs, f_range)
    return i_phase, i_amp, i_freq

# function to load bursting epochs
def get_burst_epochs(data, fs, f_range, amp_thresh_dual):
    bursting = burst.detect_bursts_dual_threshold(data, fs, f_range, amp_thresh_dual)
    bursting_stats = burst.compute_burst_stats(bursting, fs)
    return bursting, bursting_stats

# function to get psd by taking fourier transform followed by median filter
def get_psd(x, fs, Hzmed=0, zeropad=False, usehanning = False, usemedfilt = True):
    fs = int(fs)
    if zeropad:
        N = 2**(int(math.log(len(x), 2))+1)
        x = np.hstack([x,np.zeros(N-len(x))])
    else:
        N = len(x)
    f = np.arange(0,fs/2,fs/N)
    if usehanning:
        win = np.hanning(N)
        rawfft = np.fft.fft(x*win)
    else:
        rawfft = np.fft.fft(x)
    psd = np.abs(rawfft[:len(f)])**2
    if usemedfilt:
        sampmed = np.argmin(np.abs(f-Hzmed/2.0))
        psd = scsig.medfilt(psd,sampmed*2+1)
    return f, psd

# function to get PSD using periodogram
def get_psd_v2(data, fs):
    freq, psd = tsa.periodogram(data,fs,normalize=True)  
    return freq, psd

# function to run fft
def get_fft(data, fs):
    sp = np.abs(np.fft.fft(data))
    freq = np.fft.fftfreq(len(data))*fs
    # This breaks the data up into two-second windows (nperseg=fs*2) and applies a hanning window 
    # to the time-series windows (window='hann'). It then FFTs each hanning'd window, and then 
    # averages all those FFTs (method='mean') mean of spectrogram (Welch)
    freq_mean, P_mean = spectral.compute_spectrum(data, fs, method='mean', window='hann', nperseg=fs*2) 
    return freq, sp, freq_mean, P_mean

# function to get spectrogram 
def get_spectrogram(data, fs, freq_band=(0,50), norm=True, interp=True):
    f, t, Sxx = scsig.stft(data, fs, nperseg=2*fs, noverlap=fs/2.)
    if freq_band is not None:
        idx_band = np.logical_and(f >= freq_band[0], f <= freq_band[1])
        f = f[idx_band]
        Sxx = Sxx[idx_band, :]
    Sxx = np.square(np.abs(Sxx))
    if norm:
        sum_pow = Sxx.sum(0).reshape(1, -1)
        Sxx = np.divide(Sxx, sum_pow)
    return f, t, Sxx

# function to get STFT spectrogram
def get_spectrogram_stft(x, Fs, window_size, return_complex = False):    
    t_max = len(x)/float(Fs)
    t_x = np.arange(0,t_max,1/float(Fs)) 
    samp_start_spec = int(np.ceil(window_size/2))
    t_spec = t_x[samp_start_spec:samp_start_spec+len(t_x)-window_size]
    f = np.arange(0,Fs/2.,Fs/float(window_size))
    N_f = len(f)
    samp_start_max = len(x) - window_size + 1
    spec = np.zeros((samp_start_max,N_f),dtype=complex)
    for samp_start in range(samp_start_max):
        spec[samp_start] = np.fft.fft(x[samp_start:samp_start + window_size]*np.hanning(window_size))[:len(f)]
    if not return_complex:
        spec = np.abs(spec)
    return t_spec, f, spec

# function to get relative power
def get_relative_power(Sxx, f, f_band=(6,10)):
    idx = np.logical_and(f >= f_band[0], f <= f_band[1])
    rel_pow = np.divide(np.nansum(Sxx[idx],0), np.nansum(Sxx,0))
    return rel_pow

# function find power in a specific range
def calcpow(f, psd, flim):
    fidx = np.logical_and(f>=flim[0],f<=flim[1])
    return np.sum(psd[fidx])/np.float(len(f)*2)

# function get the slope of the power spectrum
def get_psd_slope(f, psd, fslopelim = (80,200), flatten_thresh = 0):
    fslopeidx = np.logical_and(f>=fslopelim[0],f<=fslopelim[1])
    slopelineF = f[fslopeidx]
    x = np.log10(slopelineF)
    y = np.log10(psd[fslopeidx])
    lm = linear_model.RANSACRegressor(random_state=42)
    lm.fit(x[:, np.newaxis], y)
    slopelineP = lm.predict(x[:, np.newaxis])
    psd_flat = y - slopelineP.flatten()
    mask = (psd_flat / psd_flat.max()) < flatten_thresh
    psd_flat[mask] = 0
    slopes = lm.estimator_.coef_
    slopes = slopes[0]
    return slopes, slopelineP, slopelineF

# function to calculate center frequencies
def centerfreq(f, psd, frange):
    frangeidx = np.logical_and(f>frange[0], f<frange[1])
    psdbeta = psd[frangeidx]
    cfs_idx = psdbeta.argmax() + np.where(frangeidx)[0][0]
    cf = f[cfs_idx]    
    return cf

# function to notch filter with center freq cs and bandwidth bw
def notch(x, cf, bw, Fs=1000, order=3):
    nyq_rate = Fs / 2.0
    f_range = [cf - bw / 2.0, cf + bw / 2.0]
    Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
    b, a = scsig.butter(order, Wn, 'bandstop')
    return scsig.filtfilt(b, a, x)

# function to remove edge artifacts
def rmvedge(x, cf, Fs, w = 3):
    N = np.int(np.floor(w * Fs / cf))
    return x[N:-N]
    
# function to calculate n:m phase-phase coupling bw 2 oscillations
# flo=(6, 10); fhi=(60,100)
# nm : n:m is the ratio of low frequency to high frequency 
# (e.g. if flo ~= 8 and fhi ~= 24, then n:m = 1:3)
def get_nm_ppcoupling(x, flo, fhi, Fs, nm=(1,3)):
    phalo, _ = pa_series(x, x, flo, flo, fs = Fs)
    phahi, _ = pa_series(x, x, fhi, fhi, fs = Fs)
    phalo, phahi = _trim_edges(phalo, phahi)
    phadiffnm = phalo*nm[1] - phahi*nm[0]
    plf = np.abs(np.mean(np.exp(1j*phadiffnm)))
    return phadiffnm, plf

# function to calculate n:m coupling for many frequencies and values of 'm' for a single signal
def get_many_nm_ppcoupling(x, floall, bw, M, Fs):
    n_flo = len(floall)
    plfs = np.zeros((n_flo,M-1))
    for f in range(n_flo):
        for midx in range(M-1):
            m = midx + 2
            fhi = (floall[f]-bw,floall[f]+bw)
            flo = (floall[f]/m-bw/m,floall[f]/m+bw/m)
            plfs[f,midx] = get_nm_ppcoupling(x, flo, fhi, (1,m),Fs)
    return plfs

# function to calculate the time-frequency representation of the signal 'x' over 
# the frequencies in 'f0s' using morlet wavelets returns time-frequency representation of signal x
def morletT(x, f0s, Fs, w = 7, s = .5):
    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')
        
    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F,T],dtype=complex)
    for f in range(F):
        mwt[f] = morletf(x, f0s[f], Fs, w = w, s = s)
    return mwt

# function to convolve a signal with a complex wavelet
# The real part is the filtered signal; np.abs() of output gives the 
# analytic amplitude and np.angle() of output gives the analytic phase
# returns complex time series; f0 : Center frequency of bandpass filter
def morletf(x, f0, Fs, w = 7, s = .5, M = None, norm = 'sss'):
    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')
    if M == None:
        M = w * Fs / f0
    morlet_f = scsig.morlet(M, w = w, s = s)
    morlet_f = morlet_f    
    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')
    mwt_real = np.convolve(x, np.real(morlet_f), mode = 'same')
    mwt_imag = np.convolve(x, np.imag(morlet_f), mode = 'same')
    return mwt_real + 1j*mwt_imag

# function to filter signal with an FIR filter but don't remove edge artifacts
def firfedge(x, f_range, fs=1000, w=3):
    nyq = np.float(fs / 2)
    Ntaps = np.floor(w * fs / f_range[0])
    taps = scsig.firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    return scsig.filtfilt(taps, [1], x)

# functiont to get window rms
def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'same'))

# function to plot lfp data
def plot_lfp(data, filt_data, times):
    fig = plt.figure(figsize=(12,4))
    plt.plot(times, data, label='Raw')
    plt.plot(times, filt_data, label='Filtered')
    plt.ylabel("Amp [uV]", fontsize=16)
    plt.xlabel("Time [s]", fontsize=16)
    plt.legend()
    return fig

# function to plot fft data
def plot_fft(p, f):
    fig = plt.figure(figsize=(12,4))
    plt.plot(f, p)
    plt.ylabel("Power [V^2/Hz]", fontsize=16)
    plt.xlabel("Freq [Hz]", fontsize=16)
    return fig

# function to plot fft data
def plot_psd(psd, f):
    fig = plt.figure(figsize=(12,4))
    plt.plot(f, psd)
    plt.ylabel("PSD [au]", fontsize=16)
    plt.xlabel("Freq [Hz]", fontsize=16)
    return fig

# function to plot instantaneous data
def plot_inst_data(data, filt_data, times, inst_phase, inst_amp, inst_freq):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12,16))
    ax1.plot(times, data, label='Raw')
    ax1.plot(times, filt_data, label='Filtered')
    ax1.set_ylabel("Amp [uV]", fontsize=16)
    ax1.set_title("Signal", fontsize=16)
    ax2.plot(times, inst_phase)
    ax2.set_ylabel("Phase [rad]", fontsize=16)
    ax2.set_title("Inst Phase", fontsize=16)
    ax3.plot(times, inst_amp)
    ax3.set_ylabel("Amp [uV]", fontsize=16)
    ax3.set_title("Inst Amplitude", fontsize=16)
    ax4.plot(times, inst_freq)
    ax4.set_xlabel("Time [s]", fontsize=16)
    ax4.set_ylabel("Freq [Hz]", fontsize=16)
    ax4.set_title("Inst Frequency", fontsize=16)
    ax4.set_ylim([0,45])
    return fig

# function to plot bursting data
def plot_bursting(data, filt_sig, times, bursting):
    bursts = np.ma.array(data, mask=np.invert(bursting))
    fig = plt.figure(figsize=(12,4))
    plt.plot(times, data, label='Raw')
    plt.plot(times, filt_sig, label='Filtered')
    plt.plot(times, bursts, label='bursts')
    plt.ylabel("Amp [uV]", fontsize=16)
    plt.xlabel("Time [s]", fontsize=16)
    plt.legend()
    return fig

# function to plot spectrogram data adn relative power
def plot_specgram(data, filt_data, times, f, t, P, P_rel):
    _extent = [min(times-times[0]), max(times), min(f), max(f)]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12,16))
    ax1.plot(times, data, label='Raw')
    ax1.plot(times, filt_data, label='Filtered')
    ax1.set_ylabel("Amp [uV]", fontsize=16)
    ax1.set_xlim([min(times), max(times)])
    ax2.imshow(P,aspect='auto',extent=_extent,origin='lower', cmap='Spectral_r')
    ax2.set_ylabel("Freq [Hz]", fontsize=16)
    ax2.set_xlim([min(t), max(t)])
    ax3.plot(t, P_rel)
    ax3.set_ylabel("Power Ratio [a.u.]", fontsize=16)
    ax3.set_xlabel("Time [s]", fontsize=16)
    ax3.set_xlim([min(t), max(t)])

# function to plot phase phase coupling 
def plot_nm_ppcoupling(plfs, floall, M, bw, clim1=(0,1)):
    plfs2 = np.zeros((len(floall)+1,M))
    plfs2[:len(floall),:M-1] = plfs
    fig = plt.figure(figsize=(5,5))
    cax = plt.pcolor(range(2,M+2), np.append(floall,100), plfs2, cmap='jet')
    cbar = plt.colorbar(cax, ticks=clim1)
    cbar.ax.set_yticklabels(clim1,size=20)
    cbar.ax.set_ylabel('Phase locking factor', size=20)
    plt.clim(clim1)
    plt.axis([2, M+1, floall[0],floall[-1]+10])
    plt.xlabel('M', size=20)
    plt.ylabel('Frequency (Hz)', size=20)
    ax = plt.gca()
    ax.set_yticks(np.array(floall)+bw)
    ax.set_yticklabels(["%d" % n for n in floall],size=20)
    plt.xticks(np.arange(2.5,M+1),["%d" % n for n in np.arange(2,M+1)],size=20)
    plt.tight_layout()
    return fig
    
# function to get less than or equal to val
def find_le(arr, val):
    i = bisect.bisect_right(arr, val)
    if i:
        return i-1, arr[i-1]
    else:
        return None, None

# function to get greater than or equal to val
def find_ge(arr, val):
    i = bisect.bisect_left(arr, val)
    if i != len(arr):
        return i, arr[i]
    else:
        return None, None

# function to get calculate significance with resampling
def resample_coupling(x1, x2, couplingfn, cfn_dict = {}, Nshuff=100, min_change=.1):
    Nsamp = len(x1)
    y_real = couplingfn(x1,x2,**cfn_dict)
    y_shuff = np.zeros(Nsamp)
    for n in range(Nshuff):
        offsetfract = min_change + (1-2*min_change)*np.random.rand()
        offsetsamp = np.int(Nsamp*offsetfract)
        x2_shuff = np.roll(x2,offsetsamp)
        y_shuff[n] = couplingfn(x1,x2_shuff,**cfn_dict)
    z = (y_real - np.mean(y_shuff)) / np.std(y_shuff)
    return _z2p(z)
    
# function to get p-val
def _z2p(z):
    p, _ = integrate.quad(lambda x: 1/np.sqrt(2*np.pi)*np.exp(-x**2/2),-np.inf,z)
    return np.min((p,1-p))

# function to get linear fit on 2d data
def linfit(x,y):
    mb = np.polyfit(x,y,1)
    xs = np.array([np.min(x),np.max(x)])
    yfit = mb[1] + xs*mb[0]
    return xs, yfit

# function to regress 1 variable out of another
def regressout(x,y):
    mb = np.polyfit(x,y,1)
    return y - mb[1] - x*mb[0]

# function to normalize data
def norm01(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))
    
# function to calculate the p-value for a pearson correlation from r and n
def pearsonp(r, n):
    if abs(r) == 1:
        return 0
    else:
        df = n-2
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        return betainc(0.5*df, 0.5, df / (df + t_squared))

# function to correlate two 2d matrix and get a correlation map matrix
def generate_correlation_map(x, y):
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

#csc_filename = 'CSCST11.ncs'
#f_int = (6,10)
#data, times, fs = load_cscdata(csc_filename)
#filt_data = get_bandpass_filter_signal(data, fs, 'bandpass', f_int)
#inst_phase, inst_amp, inst_freq = get_inst_data(data, fs, f_int)
#fig1 = plot_lfp(data, filt_data, times)
#plt.show()
#fig2 = plot_inst_data(data, filt_data, times, inst_phase, inst_amp, inst_freq)
#plt.show()
#bursting, bursting_stats = get_burst_epochs(data, fs, f_int, (1,2))
#fig3 = plot_bursting(data, filt_data, times, bursting)
#plt.show()
#f, t, P = get_spectogram(data,fs)
#P_rel = get_relative_power(P, f, f_int)
#fig4 = plot_specgram(data, filt_data, times, f, t, P, P_rel)
#f, psd = get_psd(data, fs)