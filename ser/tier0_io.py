import torch, torchaudio

TARGET_SR = 16_000

# Global cache for VAD model to avoid reloading on every call
_vad_model_cache = None
_vad_utils_cache = None

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
    global _vad_model_cache, _vad_utils_cache
    
    # Load model only once and cache it
    if _vad_model_cache is None or _vad_utils_cache is None:
        try:
            # Use source='local' to prefer local cache and avoid network calls
            _vad_model_cache, _vad_utils_cache = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', onnx=False, trust_repo=True, 
                force_reload=False, source='local'
            )
        except Exception as e:
            # If loading fails (network issue, cache issue), try with source='github' as fallback
            try:
                _vad_model_cache, _vad_utils_cache = torch.hub.load(
                    'snakers4/silero-vad', 'silero_vad', onnx=False, trust_repo=True, 
                    force_reload=False
                )
            except Exception:
                # If all else fails, return the original audio (skip VAD)
                print(f"Warning: Failed to load VAD model, skipping VAD processing: {e}")
                return wav_16k
    
    get_speech_timestamps = _vad_utils_cache[0]  # get_speech_timestamps function
    ts_list = get_speech_timestamps(wav_16k, _vad_model_cache, sampling_rate=TARGET_SR)
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
    if speech.numel() <= 1:                         # no speech or empty
        # Return minimal valid MFCC features instead of raising error
        return torch.zeros(n_mfcc, 100)  # (n_mfcc, time_frames)
    return mfcc_40(speech, n_mfcc=n_mfcc)
