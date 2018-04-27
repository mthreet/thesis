import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from scipy.signal import kaiser
from sklearn.decomposition import FastICA


def load_dataset(in_dir, sr, speaker_ids, num_wavs, num_speakers, num_samples):
    wav_data = np.zeros([num_wavs, num_speakers + 1, num_samples])
    for i in range(num_wavs):
        fdir = os.path.join(in_dir, str(i + 1))
        for j in range(num_speakers):
            fname = os.path.join(fdir, 'speaker_{}.wav'.format(speaker_ids[j]))
            wav_data[i, j], _ = librosa.core.load(fname, sr)
        fname = os.path.join(fdir, 'comb.wav')
        wav_data[i, -1], _ = librosa.core.load(fname, sr)

    return wav_data


def plot_audio(wav_data, num_plots):
    num_sigs = wav_data.shape[1]
    for i in range(num_plots):
        plt.figure(i + 1)
        for j in range(num_sigs - 1):
            plt.subplot(num_sigs, 1, j + 1)
            plt.plot(wav_data[i, j])
            plt.title('Signal {}'.format(j + 1))
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
        plt.subplot(num_sigs, 1, num_sigs)
        plt.plot(wav_data[i, -1])
        plt.title('Combined Signals')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.tight_layout()
    plt.show()


def make_spectrograms(wav_data, n_fft, hop_length):
    num_examps, num_sigs, num_samples = wav_data.shape
    spect_data = np.zeros([num_examps, num_sigs, int(n_fft / 2) + 1,
                           int((num_samples-num_samples %
                                hop_length)/hop_length+1)],
                          dtype=np.complex)
    for i in range(num_examps):
        for j in range(num_sigs):
            spect_data[i, j] = librosa.core.stft(wav_data[i, j], n_fft,
                                                 hop_length, n_fft,
                                                 kaiser(n_fft, 14),
                                                 center=True)
    return spect_data


def plot_spects(spect_data, num_plots, sr):
    num_sigs = spect_data.shape[1]
    for i in range(num_plots):
        for j in range(num_sigs - 1):
            plt.figure(i + 1)
            plt.subplot(num_sigs, 1, j + 1)
            librosa.display.specshow(
                librosa.amplitude_to_db(np.abs(spect_data[i, j])), sr=sr)
            plt.title('Spectrogram of Signal {}'.format(j + 1))
            plt.xlabel('Time')
            plt.ylabel('Frequency')
        plt.subplot(num_sigs, 1, num_sigs)
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spect_data[i, -1])), sr=sr)
        plt.title('Spectrogram of Combined Signal')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
    plt.show()


def nnmf(spect_data):
    num_examps, num_speakers, f, t = spect_data.shape
    num_speakers -= 1
    W = np.zeros([num_examps, f, num_speakers])
    H = np.zeros([num_examps, num_speakers, t])
    recon_spects = np.zeros([num_examps, num_speakers + 1, f, t],
                            dtype=np.complex)
    for i in range(num_examps):
        W[i], H[i] = librosa.decompose.decompose(np.abs(spect_data[i, -1]),
                                                 num_speakers)
        phase = np.angle(spect_data[i, -1])
        for j in range(num_speakers):
            recon_spects[i, j] = W[i, :, None, j].dot(
                H[i, j, None, :]) * np.exp(1j * phase)
        recon_spects[i, -1] = W[i].dot(H[i])
    return W, H, recon_spects


def ica_decomp(spect_data):
    num_examps, num_speakers, f, t = spect_data.shape
    num_speakers -= 1
    recon_spects = np.zeros(spect_data.shape)
    ica_tran = FastICA(n_components=num_speakers)
    for i in range(num_examps):
        sep_sigs = ica_tran.fit_transform(spect_data[i,-1])
        for j in range(num_speakers):
            recon_spects[i,j] = sep_sigs[j]
        recon_spects[i,-1] = np.sum(recon_spects[i,:-1, :], axis=0)
    return recon_spects

def spect_to_audio(recon_spects, n_fft, hop_length):
    num_examps, num_speakers, f, t = recon_spects.shape
    num_speakers -= 1
    recon_wavs = np.zeros([num_examps, num_speakers + 1,
                           int((t-1) * hop_length)])
    for i in range(num_examps):
        for j in range(num_speakers+1):
            recon_wavs[i, j] = librosa.core.istft(recon_spects[i,j],
                                                  hop_length, n_fft,
                                                  kaiser(n_fft,14),
                                                  center=True)
    return recon_wavs


def plot_orig_vs_recon(wav_data, recon_wavs, num_plots):
    num_sigs = wav_data.shape[1]
    for i in range(num_plots):
        plt.figure(i + 1)
        for j in range(num_sigs - 1):
            plt.subplot(num_sigs, 2, 2*j+1)
            plt.plot(wav_data[i, j])
            plt.ylim([-1, 1])
            plt.title('Original Signal {}'.format(j+1))
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.subplot(num_sigs, 2, 2*j+2)
            plt.plot(recon_wavs[i, j])
            plt.ylim([-1, 1])
            plt.title('Reconstructed Signal {}'.format(j+1))
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
        plt.subplot(num_sigs, 2, 2*num_sigs-1)
        plt.plot(wav_data[i, -1])
        plt.ylim([-1, 1])
        plt.title('Original Combined Signals')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.subplot(num_sigs, 2, 2*num_sigs)
        plt.plot(np.sum(recon_wavs[i,:-1, :], axis=0))
        plt.ylim([-1, 1])
        plt.title('Reconstructed Combined Signals')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.tight_layout()
    plt.show()


def compute_recon_scores(wav_data, recon_wavs):
    num_examps, num_sigs, num_samples = recon_wavs.shape
    # Number of samples in recon_wavs is aways <= number of samples in wav_data
    wav_data = wav_data[:,:,:num_samples]
    out = np.empty(recon_wavs[:,:-1,:].shape)
    out[:] = np.nan
    mse = np.mean(np.square(wav_data[:,:-1,:] - recon_wavs[:, :-1, :]))
    mape =  np.nanmean(np.abs(np.divide(wav_data[:,:-1,:] - recon_wavs[:,:-1,:],
                                       wav_data[:,:-1,:], out=out,
                                       where=wav_data[:,:-1,:]!=0)))
    return mse, mape


def save_wavs(wav_data, recon_wavs, out_dir, sr):
    num_examps, num_sigs, num_samples = recon_wavs.shape
    # Number of samples in recon_wavs is aways <= number of samples in wav_data
    wav_data = wav_data[:, :, :num_samples]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(num_examps):
        librosa.output.write_wav(os.path.join(out_dir,
                                              'comb_{}.wav'.format(i)),
                                 wav_data[i,-1],
                                 sr=sr)
        librosa.output.write_wav(os.path.join(out_dir,
                                              'recon_{}.wav'.format(i)),
                                 recon_wavs[i, 0],
                                 sr=sr)
        librosa.output.write_wav(os.path.join(out_dir,
                                              'true_{}.wav'.format(i)),
                                 wav_data[i, 0],
                                 sr=sr)
