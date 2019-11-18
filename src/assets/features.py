import torch

# compute features in pytorch

def compute_features(audio, hp):

    # compute spectrogram from first channel
    stft = torch.stft(audio, hp.features.n_fft, hp.features.hop, hp.features.win_len, center=False)



    return