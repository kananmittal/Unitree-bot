import torch, torchaudio

TARGET_SR = 16_000

# ---------- load/resample ----------
def load_audio(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)                # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)        # mono
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    wav = wav.squeeze(0)                           # (T,)
    wav = wav / (wav.abs().max().clamp_min(1e-8))  # normalize
    return wav

# ---------- denoise (Demucs DNS64) ----------
def denoise_demucs(wav_16k: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    from denoiser import pretrained
    model = pretrained.dns64().to(device).eval()
    with torch.no_grad():
        x = wav_16k.to(device)[None, None, :]      # (1,1,T)
        y = model(x).squeeze().cpu()               # (T,)
    return y

# ---------- VAD (Silero) → keep only speech ----------
def vad_silero_keep_speech(wav_16k: torch.Tensor) -> torch.Tensor:
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False, trust_repo=True)
    get_speech_timestamps = utils[0]  # get_speech_timestamps function
    ts_list = get_speech_timestamps(wav_16k, model, sampling_rate=TARGET_SR)
    if not ts_list:
        return torch.zeros(1)
    chunks = [wav_16k[t['start']:t['end']] for t in ts_list]
    return torch.cat(chunks)

# ---------- MFCC ----------
def mfcc_40(wav_16k: torch.Tensor, n_mfcc: int = 40) -> torch.Tensor:
    t = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SR,
        n_mfcc=n_mfcc,
        melkwargs=dict(n_fft=512, hop_length=160, win_length=400,
                       f_min=20.0, f_max=TARGET_SR/2, window_fn=torch.hamming_window)
    )
    mfcc = t(wav_16k.unsqueeze(0)).squeeze(0)      # (C, T)
    return mfcc

# ---------- end-to-end Tier 0 ----------
def tier0_to_mfcc(path: str, device: str = "cpu", n_mfcc: int = 40) -> torch.Tensor:
    wav = load_audio(path)
    den = denoise_demucs(wav, device=device)
    speech = vad_silero_keep_speech(den)
    if speech.numel() == 1:                         # no speech
        return torch.zeros(n_mfcc, 1)
    return mfcc_40(speech, n_mfcc=n_mfcc)
