"""
audio_input.py
--------------
Audio capture for the AI Voice Translator.

Backend: PyAudioWPatch (a PyAudio fork with Windows WASAPI *loopback* support).
This single backend gives us both:
    * Microphone capture
    * System / app audio capture (loopback = "whatever is playing on speakers")

What this module does:
    1. Lists selectable input devices (mics + loopback).
    2. Opens a capture stream on a background thread.
    3. Runs a simple energy-based VAD (Voice Activity Detection) to split the
       stream into "utterances" (one spoken sentence between pauses).
    4. Calls a callback with each finished utterance as 16-bit mono PCM, ready
       for speech-to-text. Also reports the live audio level for the GUI meter.

Why VAD? We translate sentence-by-sentence: start collecting when sound rises
above a threshold, and finish the utterance after a short silence. This is what
keeps things "near real-time" without cutting words in half.
"""

import threading

import numpy as np
import pyaudiowpatch as pyaudio

import config


def list_input_devices():
    """
    Return selectable inputs as [{"label","id","channels","rate","kind"}, ...].

    Loopback devices (system audio) are listed first because that is the
    primary source the user chose. Each entry carries the channel count and
    sample rate we must open the device with (WASAPI shared mode requires
    matching the device's native format).
    """
    devices = []
    pa = pyaudio.PyAudio()
    try:
        # --- System audio (loopback) first ---------------------------------
        try:
            for lb in pa.get_loopback_device_info_generator():
                devices.append({
                    "label": f"System audio: {lb['name'].replace(' [Loopback]', '')}",
                    "id": int(lb["index"]),
                    "channels": int(lb["maxInputChannels"]) or 2,
                    "rate": int(lb["defaultSampleRate"]),
                    "kind": "loopback",
                })
        except Exception:
            pass  # no loopback available on this machine

        # --- Real microphones (WASAPI inputs that are not loopback) --------
        try:
            wasapi_index = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        except Exception:
            wasapi_index = None

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) <= 0:
                continue
            if wasapi_index is not None and info.get("hostApi") != wasapi_index:
                continue
            if info.get("isLoopbackDevice", False):
                continue
            devices.append({
                "label": f"Microphone: {info['name']}",
                "id": int(info["index"]),
                "channels": int(info["maxInputChannels"]),
                "rate": int(info["defaultSampleRate"]),
                "kind": "mic",
            })
    finally:
        pa.terminate()

    if not devices:
        devices.append({"label": "(no input devices found)", "id": None,
                        "channels": 1, "rate": config.SAMPLE_RATE, "kind": "none"})
    return devices


def _find_device(device_id):
    """Look up a device entry by its id, or None."""
    for d in list_input_devices():
        if d["id"] == device_id:
            return d
    return None


def normalize_audio_samples(samples):
    """Apply bounded gain to quiet int16 audio before sending it to STT.

    System/loopback devices often deliver dialogue at a small fraction of full
    scale. Scaling only the captured chunk improves recognizer input quality;
    the VAD timing and the number of network requests remain unchanged.
    """
    samples = np.asarray(samples, dtype=np.int16)
    if samples.size == 0:
        return samples

    # Convert before abs() so the int16 -32768 edge case cannot overflow.
    peak = float(np.max(np.abs(samples.astype(np.float32))) / 32768.0)
    target_peak = float(getattr(config, "AUDIO_NORMALIZE_PEAK", 0.0))
    max_gain = float(getattr(config, "AUDIO_NORMALIZE_MAX_GAIN", 1.0))
    if peak <= 0.0005 or peak >= target_peak or max_gain <= 1.0:
        return samples

    gain = min(target_peak / peak, max_gain)
    if gain <= 1.01:
        return samples

    scaled = samples.astype(np.float32) * gain
    return np.clip(scaled, -32768, 32767).astype(np.int16)


class AudioCapture:
    """
    Captures audio on a background thread and emits complete utterances.

    on_utterance(pcm_bytes, sample_rate): called with 16-bit MONO PCM bytes for
        each detected sentence. Keep it fast (just enqueue) - it runs on the
        capture thread.
    on_level(rms, is_speech): called ~10x/sec with the current audio level
        (0..1) so the GUI can show a live indicator. Optional.
    """

    def __init__(self, on_utterance, on_level=None, on_interim=None,
                 on_utterance_with_id=None):
        self.on_utterance = on_utterance
        self.on_level = on_level
        self.on_interim = on_interim
        # Optional extended callback used by the desktop pipeline to order
        # provisional captions without breaking the original two-argument API.
        self.on_utterance_with_id = on_utterance_with_id

        self._thread = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()   # set => discard audio (e.g. while TTS speaks)
        self._error = None

    # ------------------------------------------------------------------ control
    def start(self, device_id):
        """Open the given device and begin capturing on a background thread."""
        device = _find_device(device_id) if device_id is not None else None
        if device is None:
            # Fall back to the first available input device.
            candidates = list_input_devices()
            device = candidates[0] if candidates else None
        if device is None or device["id"] is None:
            raise RuntimeError("No usable audio input device was found.")

        self._stop_event.clear()
        self._paused.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, args=(device,), daemon=True, name="AudioCapture")
        self._thread.start()

    def stop(self):
        """Signal the capture thread to stop and wait briefly for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def pause(self):
        """Discard incoming audio (used to avoid capturing our own TTS)."""
        self._paused.set()

    def resume(self):
        self._paused.clear()

    @property
    def error(self):
        return self._error

    # ------------------------------------------------------------------ worker
    def _run(self, device):
        """Capture loop (background thread). Reads blocks, runs VAD, emits."""
        pa = pyaudio.PyAudio()
        stream = None
        rate = device["rate"]
        channels = device["channels"]
        block = max(1, int(rate * config.BLOCK_DURATION))

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device["id"],
                frames_per_buffer=block,
            )
        except Exception as exc:
            # Surface a friendly message; the pipeline/GUI can display it.
            self._error = f"Could not open audio device: {exc}"
            if stream is not None:
                stream.close()
            pa.terminate()
            return

        # VAD state
        speech_buffer = []          # list of mono int16 numpy arrays
        silence_seconds = 0.0
        speech_seconds = 0.0
        interim_seconds = 0.0
        in_speech = False
        utterance_id = 0
        next_utterance_id = 0
        noise_floor = 0.0

        try:
            while not self._stop_event.is_set():
                try:
                    raw = stream.read(block, exception_on_overflow=False)
                except Exception as exc:
                    self._error = f"Audio read failed: {exc}"
                    break

                # Convert interleaved int16 -> mono int16 via numpy.
                samples = np.frombuffer(raw, dtype=np.int16)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

                # If paused (e.g. TTS is speaking), throw the audio away and
                # reset VAD so we don't translate our own voice.
                if self._paused.is_set():
                    speech_buffer, silence_seconds, speech_seconds = [], 0.0, 0.0
                    interim_seconds = 0.0
                    in_speech = False
                    utterance_id = 0
                    noise_floor = 0.0
                    if self.on_level:
                        self.on_level(0.0, False)
                    continue

                # RMS energy in 0..1 range (int16 max is 32768).
                rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2))) \
                    if samples.size else 0.0

                noise_floor = update_noise_floor(noise_floor, rms)
                is_loud = rms >= adaptive_vad_threshold(rms, noise_floor)
                if self.on_level:
                    self.on_level(rms, is_loud)

                if is_loud:
                    if not in_speech:
                        next_utterance_id += 1
                        utterance_id = next_utterance_id
                        interim_seconds = 0.0
                    in_speech = True
                    silence_seconds = 0.0
                    speech_seconds += config.BLOCK_DURATION
                    interim_seconds += config.BLOCK_DURATION
                    speech_buffer.append(samples)
                    interim_every = float(getattr(
                        config, "VAD_INTERIM_SECONDS", 0.0))
                    if self.on_interim and interim_every > 0 and \
                            interim_seconds >= interim_every:
                        self._emit(speech_buffer, rate, interim=True,
                                   utterance_id=utterance_id)
                        interim_seconds = 0.0
                elif in_speech:
                    # Trailing silence: keep a little audio, count the gap.
                    silence_seconds += config.BLOCK_DURATION
                    speech_buffer.append(samples)

                # End of utterance? enough silence after real speech.
                end_by_silence = (in_speech
                                  and silence_seconds >= config.VAD_SILENCE_SECONDS
                                  and speech_seconds >= config.VAD_MIN_SPEECH_SECONDS)
                end_by_length = speech_seconds >= config.VAD_MAX_UTTERANCE_SECONDS

                if end_by_silence or end_by_length:
                    if end_by_length and not end_by_silence:
                        combined = np.concatenate(speech_buffer)
                        cut = quietest_cut_index(combined, rate)
                        self._emit([combined[:cut]], rate,
                                   utterance_id=utterance_id)
                        carry = combined[cut:]
                        if carry.size:
                            # Continue with the post-boundary audio under a new
                            # id so late interim replies cannot replace its final.
                            next_utterance_id += 1
                            utterance_id = next_utterance_id
                            speech_buffer = [carry]
                            speech_seconds = carry.size / rate
                            silence_seconds = 0.0
                            interim_seconds = 0.0
                            in_speech = True
                            continue
                    else:
                        self._emit(speech_buffer, rate,
                                   utterance_id=utterance_id)
                    speech_buffer, silence_seconds, speech_seconds = [], 0.0, 0.0
                    interim_seconds = 0.0
                    utterance_id = 0
                    in_speech = False
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    def _emit(self, buffer, rate, interim=False, utterance_id=0):
        """Join buffered blocks and hand a final or provisional PCM chunk onward."""
        if not buffer:
            return
        mono = normalize_audio_samples(np.concatenate(buffer))
        try:
            if interim:
                if self.on_interim:
                    self.on_interim(mono.tobytes(), rate, utterance_id)
            elif self.on_utterance_with_id:
                self.on_utterance_with_id(mono.tobytes(), rate, utterance_id)
            else:
                self.on_utterance(mono.tobytes(), rate)
        except Exception:
            # A failing consumer must not kill the capture thread.
            pass


def quietest_cut_index(samples, sample_rate, search_seconds=None):
    """Find a recent low-energy boundary to avoid splitting words on force-cut."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < 2:
        return samples.size
    window = max(1, int(round(0.03 * sample_rate)))
    search_seconds = (getattr(config, "VAD_CUT_SEARCH_SECONDS", 1.2)
                      if search_seconds is None else search_seconds)
    search_start = max(0, samples.size - int(sample_rate * search_seconds))
    if samples.size - search_start < window * 2:
        return samples.size
    power = samples * samples
    cumulative = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    energies = cumulative[window:] - cumulative[:-window]
    first = min(max(0, search_start), len(energies) - 1)
    last = max(first + 1, len(energies))
    start = first + int(np.argmin(energies[first:last]))
    return min(samples.size, start + window // 2)


def adaptive_vad_threshold(rms, noise_floor):
    """Return a speech threshold that follows background audio safely."""
    absolute = float(getattr(config, "VAD_ENERGY_THRESHOLD", 0.01))
    multiplier = float(getattr(config, "VAD_NOISE_MULTIPLIER", 1.0))
    return max(absolute, float(noise_floor) * multiplier)


def update_noise_floor(noise_floor, rms):
    """Update the slowly moving RMS background estimate used by the VAD."""
    rms = max(0.0, float(rms))
    if noise_floor <= 0.0:
        # Do not let the first loud speech block calibrate the floor upward.
        return min(rms, float(getattr(config, "VAD_ENERGY_THRESHOLD", 0.01)))
    if rms < noise_floor:
        return noise_floor * 0.85 + rms * 0.15
    # Rise slowly so a long sentence cannot promote itself to background noise.
    return noise_floor * 0.995 + rms * 0.005
