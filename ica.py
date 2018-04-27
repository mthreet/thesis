from utils import load_dataset, plot_audio, make_spectrograms, plot_spects, \
    plot_orig_vs_recon, spect_to_audio, compute_recon_scores, ica_decomp

in_dir = '/home/mthret/class/thesis/data/mf'
sr = 16000
speaker_ids = ['8', '16']
num_wavs = 2
num_speakers = 2
num_samples = 32000

n_fft = 512
hop_length = int(n_fft/4)

wav_data = load_dataset(in_dir, sr, speaker_ids, num_wavs, num_speakers,
                        num_samples)

# plot_audio(wav_data, 2)

spect_data = make_spectrograms(wav_data, n_fft, hop_length)

# plot_spects(spect_data, 2, sr)

recon_spects = ica_decomp(spect_data)

# plot_spects(recon_spects, 2, sr)

recon_wavs = spect_to_audio(recon_spects, n_fft, hop_length)

plot_orig_vs_recon(wav_data, recon_wavs, 2)

mse, mape = compute_recon_scores(wav_data, recon_wavs)

print('MSE of {}\nMAPE of {}'.format(mse, mape))