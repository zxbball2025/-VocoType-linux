"""Local speaker verification for the IBus dictation path.

A voice embedding is sensitive: store it in the user's private directory, and
never log or transmit the embedding. This gate rejects nonmatching windows; it
does not separate overlapping voices.
"""
from pathlib import Path
import struct
import threading
import numpy as np

PROFILE = Path.home() / '.local/share/vocotype/private/speaker-profile.bin'
MODEL = Path.home() / 'helo-voice-speaker/campplus.onnx'
RATE = 16000
THRESHOLD = 0.5
_extractor = None
_lock = threading.Lock()

class SpeakerFilterError(Exception):
    pass

def _profile():
    try:
        blob = PROFILE.read_bytes()
        if len(blob) < 4:
            raise ValueError('empty')
        count = struct.unpack('>i', blob[:4])[0]
        if count < 16 or count > 4096 or len(blob) != 4 + 4 * count:
            raise ValueError('invalid')
        vector = np.frombuffer(blob, dtype='>f4', offset=4).astype(np.float32)
        if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
            raise ValueError('invalid')
        return vector
    except (OSError, ValueError, struct.error) as exc:
        raise SpeakerFilterError('声纹不可用，请检查声纹设置') from exc

def _embedding(audio):
    global _extractor
    try:
        import sherpa_onnx
        with _lock:
            if _extractor is None:
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=1, debug=False)
                _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            stream = _extractor.create_stream()
            stream.accept_waveform(RATE, np.asarray(audio, dtype=np.float32) / 32768.0)
            stream.input_finished()
            if not _extractor.is_ready(stream):
                return None
            return np.asarray(_extractor.compute(stream), dtype=np.float32)
    except Exception as exc:
        raise SpeakerFilterError('本地声纹模型不可用') from exc

def _matches(profile, embedding):
    if embedding is None or embedding.shape != profile.shape or not np.isfinite(embedding).all():
        return False
    denominator = float(np.linalg.norm(profile) * np.linalg.norm(embedding))
    return denominator > 1e-8 and float(np.dot(profile, embedding) / denominator) >= THRESHOLD

def filter_speech(samples):
    """Verify one complete push-to-talk utterance and return it unchanged."""
    profile = _profile()
    audio = np.asarray(samples, dtype=np.int16).reshape(-1)
    if len(audio) < RATE // 2:
        raise SpeakerFilterError('语音太短，请再说一遍')
    if not _matches(profile, _embedding(audio)):
        raise SpeakerFilterError('未识别到已录入的声音，请再说一遍')
    return audio
