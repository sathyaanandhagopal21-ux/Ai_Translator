/* ==========================================================================
   AI Voice Translator - browser client
   --------------------------------------------------------------------------
   Responsibilities (all client-side, so the level meter feels instant):
     1. Capture audio from the chosen source:
          mic    - getUserMedia (your voice)
          system - getDisplayMedia (whatever is playing on this PC; Chrome/Edge)
          both   - the two mixed into one mono signal
     2. Compute a live RMS level meter every audio block.
     3. Voice-Activity Detection (VAD): split the stream into utterances by
        energy + a short trailing silence (mirrors the Python desktop VAD).
     4. Send each finished utterance (raw 16-bit PCM) to the Flask server,
        which returns the transcript + translation.
     5. Optionally fetch spoken audio (gTTS mp3) and play it here, pausing
        capture while it plays so we don't re-hear our own translation.
   ========================================================================== */

(() => {
  "use strict";

  const CFG = window.APP_CONFIG || {};
  const VAD = CFG.vad || {};
  const AUD = CFG.audio || {};
  const TH = VAD.energyThreshold ?? 0.01;
  const SILENCE_SEC = VAD.silenceSeconds ?? 0.25;
  const MIN_SPEECH_SEC = VAD.minSpeechSeconds ?? 0.2;
  const MAX_UTTER_SEC = VAD.maxUtteranceSeconds ?? 6;
  const INTERIM_SEC = VAD.interimSeconds ?? 1.5;
  const CUT_SEARCH_SEC = VAD.cutSearchSeconds ?? 1.2;
  const NOISE_MULT = VAD.noiseMultiplier ?? 2.2;
  const NORM_PEAK = AUD.normalizePeak ?? 0.85;
  const NORM_MAX_GAIN = AUD.normalizeMaxGain ?? 12;
  const MAX_CAPTION_LINES = 2;      // how many finished lines stay on screen

  // Palette (kept in sync with style.css / config.COLORS) for the meter+status.
  const COL = {
    start: "#4ade80", stop: "#fb7185", accent: "#7ee7ff", accent2: "#a78bfa",
    muted: "#8b91b0", idle: "#5b6488", text: "#f5f7ff", levelbg: "#242844",
  };

  // ---- DOM ----------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const sourceSel = $("sourceSel"), targetSel = $("targetSel"), micSel = $("micSel");
  const audioSrcSel = $("audioSrcSel"), micField = $("micField"), srcHint = $("srcHint");
  const startBtn = $("startBtn"), stopBtn = $("stopBtn"), clearBtn = $("clearBtn");
  const refreshBtn = $("refreshBtn"), speakChk = $("speakChk"), speakNote = $("speakNote");
  const badge = $("badge"), badgeText = $("badgeText");
  const detectedVal = $("detectedVal"), statusVal = $("statusVal"), voiceVal = $("voiceVal");
  const originalText = $("originalText"), subtitleText = $("subtitleText");
  const meter = $("meter"), notice = $("notice"), ttsPlayer = $("ttsPlayer");

  // ---- state --------------------------------------------------------------
  let running = false;
  let speaking = false;          // true while translated audio plays (pause VAD)
  let quietUntil = 0;            // brief cooldown after playback (speaker tail)
  let audioCtx = null, proc = null, muteNode = null, mixer = null;
  let streams = [], srcNodes = [];
  let RATE = CFG.sampleRate || 16000;
  let noiseFloor = 0;            // running estimate of the background level
  const vad = { inSpeech: false, speechSamples: 0, silenceSamples: 0,
                totalSamples: 0, buffer: [],
                uid: 0, part: 0, interimAt: 0 };

  // Chunks are processed in parallel by the server, so replies can arrive out
  // of order. These track what has already been shown so a slow, older reply
  // can never overwrite a newer one.
  let uidCounter = 0;         // increments per utterance
  let lastFinalUid = 0;       // newest utterance whose FINAL result is shown
  let lastInterimKey = 0;     // newest interim (uid*1000+part) shown
  let interimOn = INTERIM_SEC > 0;
  let interimFails = 0;       // consecutive interim errors -> back off

  // The caption is a small rolling transcript, like real subtitles: the last
  // few finished lines, plus the in-progress line while someone is still
  // talking. Without this, a finished line would blank the screen until the
  // next one arrived.
  const capLines = [], origLines = [];
  let capInterim = "", origInterim = "";

  // What each audio source can honestly do, shown under the selector.
  const HINTS = {
    mic: "Captures your voice from the microphone chosen above.",
    system: "Captures whatever is playing on this PC — a video, a call, a phone " +
            "mirrored to the PC. Needs Chrome or Edge: in the share dialog pick " +
            "“Entire Screen” and tick “Also share system audio” (sharing a single " +
            "browser tab carries that tab's audio too).",
    both: "Captures the microphone and this PC's audio together — useful for a " +
          "live two-way conversation.",
  };

  // ---- small UI helpers ---------------------------------------------------
  const enc = encodeURIComponent;

  function setBadge(active, text) {
    badge.classList.toggle("live", active);
    badge.classList.toggle("idle", !active);
    badgeText.textContent = text;
  }
  function setStatus(text, kind) {
    statusVal.textContent = text;
    statusVal.classList.remove("muted");
    statusVal.style.color = COL[kind] || COL.text;
  }
  function setVoice(text, kind) {
    voiceVal.textContent = text;
    voiceVal.style.color = COL[kind] || COL.text;
  }
  function setDetected(name) {
    detectedVal.textContent = name || "—";
    detectedVal.style.color = COL.text;
  }

  /**
   * Paint a rolling caption: the last MAX_CAPTION_LINES committed lines, then
   * the provisional line (dimmed) if speech is still in progress.
   */
  function paint(el, lines, interimLine, placeholder) {
    if (!lines.length && !interimLine) {
      el.textContent = placeholder;
      el.classList.add("placeholder");
      return;
    }
    el.classList.remove("placeholder");
    el.textContent = "";
    const frag = document.createDocumentFragment();
    for (const line of lines) {
      const d = document.createElement("div");
      d.className = "cap-line";
      d.textContent = line;
      frag.appendChild(d);
    }
    if (interimLine) {
      const d = document.createElement("div");
      d.className = "cap-line interim";
      d.textContent = interimLine;
      frag.appendChild(d);
    }
    el.appendChild(frag);
  }

  function renderCaptions() {
    paint(originalText, origLines, origInterim, "Waiting for speech…");
    paint(subtitleText, capLines, capInterim, "Translation will appear here…");
  }

  function commitLine(original, translated) {
    capInterim = origInterim = "";
    if (translated) capLines.push(translated);
    if (original) origLines.push(original);
    while (capLines.length > MAX_CAPTION_LINES) capLines.shift();
    while (origLines.length > MAX_CAPTION_LINES) origLines.shift();
    renderCaptions();
  }

  function showInterim(original, translated) {
    capInterim = translated || "";
    origInterim = original || "";
    renderCaptions();
  }

  function clearCaptions() {
    capLines.length = 0; origLines.length = 0;
    capInterim = origInterim = "";
    renderCaptions();
  }
  function showNotice(msg, kind) {
    notice.textContent = msg;
    notice.className = "notice " + (kind || "info");
    notice.hidden = false;
  }
  function hideNotice() { notice.hidden = true; }

  // ---- level meter --------------------------------------------------------
  const NUM = 28;
  const segs = [];
  function buildMeter() {
    meter.innerHTML = "";
    segs.length = 0;
    for (let i = 0; i < NUM; i++) {
      const d = document.createElement("div");
      d.className = "seg";
      meter.appendChild(d);
      segs.push(d);
    }
  }
  function segColor(frac, speech) {
    if (!speech) return COL.idle;
    if (frac < 0.6) return COL.start;
    if (frac < 0.85) return COL.accent;
    return COL.stop;
  }
  function updateMeter(rms, speech) {
    // rms is 0..1 but speech sits low (~0.02-0.15); sqrt + gain makes it visible.
    const level = Math.min(1, Math.sqrt(Math.max(0, rms)) * 2.4);
    const lit = Math.round(level * NUM);
    for (let i = 0; i < NUM; i++)
      segs[i].style.background = i < lit ? segColor(i / (NUM - 1), speech) : COL.levelbg;
  }
  function resetMeter() { for (const s of segs) s.style.background = COL.levelbg; }

  // ---- populate selectors -------------------------------------------------
  function fillSelect(sel, items, def) {
    sel.innerHTML = "";
    (items || []).forEach((name) => {
      const o = document.createElement("option");
      o.value = name; o.textContent = name;
      sel.appendChild(o);
    });
    if (def) sel.value = def;
  }
  async function populateMics() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter((d) => d.kind === "audioinput");
      const cur = micSel.value;
      micSel.innerHTML = "";
      if (!mics.length) {
        const o = document.createElement("option");
        o.value = ""; o.textContent = "(no microphone found)";
        micSel.appendChild(o); return;
      }
      mics.forEach((d, i) => {
        const o = document.createElement("option");
        o.value = d.deviceId;
        // Labels are blank until mic permission is granted (browser privacy).
        o.textContent = d.label || `Microphone ${i + 1}`;
        micSel.appendChild(o);
      });
      if (cur) micSel.value = cur;
    } catch (_) { /* ignore */ }
  }

  // ---- audio math ---------------------------------------------------------
  function computeRMS(buf) {
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
  }
  function floatTo16BitPCM(float32) {
    const view = new DataView(new ArrayBuffer(float32.length * 2));
    for (let i = 0; i < float32.length; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); // little-endian
    }
    return view.buffer;
  }

  /**
   * Scale a chunk up so its loudest sample sits near full scale.
   *
   * Dialogue in a video or a call is often mixed quiet. A chunk peaking at,
   * say, 0.06 only uses ~6% of the 16-bit range, and the recognizer does
   * noticeably worse on it - this is a common cause of wrong words from system
   * audio, where (unlike the microphone) no automatic gain control is applied.
   *
   * The gain is capped, and near-silence is left alone, so a quiet passage is
   * never amplified into hiss that the recognizer then "hears" words in.
   */
  function normalize(samples) {
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const a = samples[i] < 0 ? -samples[i] : samples[i];
      if (a > peak) peak = a;
    }
    if (peak <= 0.0005 || peak >= NORM_PEAK) return samples;
    const gain = Math.min(NORM_PEAK / peak, NORM_MAX_GAIN);
    if (gain <= 1.01) return samples;
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) out[i] = samples[i] * gain;
    return out;
  }

  /**
   * Index of the quietest moment within the last `searchLen` samples.
   *
   * Used when a long unbroken stretch has to be cut. Slicing at an arbitrary
   * instant lands mid-word and the recognizer mangles the word on both sides of
   * the boundary; that is the single biggest source of garbled text from
   * continuous audio like a video. Speech still has short gaps *between* words
   * even with no real pause, so we look back over roughly a second and cut in
   * the quietest 30 ms window we find. The outgoing chunk then ends on a gap
   * (which also helps the recognizer decide the phrase is over), and the next
   * chunk starts cleanly on the following word.
   */
  function quietestPoint(samples, searchLen) {
    const n = samples.length;
    const win = Math.max(16, Math.round(0.03 * RATE));   // 30 ms
    const from = Math.max(win, n - searchLen);
    if (n - from < win * 2) return n;      // too little audio to search
    let energy = 0;
    for (let i = from - win; i < from; i++) energy += samples[i] * samples[i];
    let best = energy, bestEnd = from;
    for (let i = from; i < n; i++) {
      energy += samples[i] * samples[i] - samples[i - win] * samples[i - win];
      if (energy < best) { best = energy; bestEnd = i; }
    }
    return bestEnd;
  }

  // ---- capture ------------------------------------------------------------
  /** The microphone, via getUserMedia. */
  async function getMicStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw new Error("no-mic-api");

    const audio = {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    };
    const dev = micSel.value;
    if (dev) audio.deviceId = { exact: dev };

    try {
      return await navigator.mediaDevices.getUserMedia({ audio });
    } catch (err) {
      // A specific device may have vanished; retry with the default mic.
      if (!dev) throw err;
      delete audio.deviceId;
      return await navigator.mediaDevices.getUserMedia({ audio });
    }
  }

  /**
   * System / app audio, via getDisplayMedia (the screen-share API).
   * This is the ONLY way a browser can hear other apps, and it is deliberate:
   * the user must pick a surface and opt into sharing its audio. Chrome/Edge on
   * Windows can share the whole machine's audio; any Chromium can share a tab's.
   */
  async function getSystemStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia)
      throw new Error("no-display-api");

    // Chrome refuses audio-only display capture, so a video track must be
    // requested. We never render it; frameRate 1 keeps the cost negligible.
    const s = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 1, displaySurface: "monitor" },
      audio: {
        echoCancellation: false,      // this is media, not a voice call - leave
        noiseSuppression: false,      // it untouched so STT gets clean audio
        autoGainControl: false,
        suppressLocalAudioPlayback: false,   // user keeps hearing it normally
      },
    });

    if (!s.getAudioTracks().length) {
      s.getTracks().forEach((t) => t.stop());
      throw new Error("no-system-audio");    // shared a surface, but muted
    }
    // The video track is kept (never displayed): in Chrome the audio track
    // belongs to the same capture session and stopping the video can kill it.
    // Keeping it also leaves Chrome's "Stop sharing" bar visible - an honest,
    // always-available way for the user to end capture.
    return s;
  }

  async function startCapture() {
    const mode = audioSrcSel.value;
    streams = [];
    srcNodes = [];

    // System audio FIRST: getDisplayMedia needs the fresh click (user
    // activation), which an awaited getUserMedia prompt would consume.
    if (mode === "system" || mode === "both") streams.push(await getSystemStream());
    if (mode === "mic" || mode === "both") streams.push(await getMicStream());

    const AC = window.AudioContext || window.webkitAudioContext;
    try { audioCtx = new AC({ sampleRate: CFG.sampleRate }); }
    catch (_) { audioCtx = new AC(); }
    if (audioCtx.state === "suspended") await audioCtx.resume();
    RATE = audioCtx.sampleRate;   // we tell the server whatever we really got

    // Mix every source down to a single mono signal for the VAD + meter.
    // System audio is usually stereo, so force an explicit mono downmix.
    mixer = audioCtx.createGain();
    mixer.gain.value = 1;
    mixer.channelCount = 1;
    mixer.channelCountMode = "explicit";
    mixer.channelInterpretation = "speakers";

    for (const s of streams) {
      const n = audioCtx.createMediaStreamSource(s);
      n.connect(mixer);
      srcNodes.push(n);
    }

    // ScriptProcessorNode is deprecated but universally supported and the
    // simplest way to read raw PCM. 2048 frames (~43ms @48kHz) rather than
    // 4096: the VAD can only notice silence at block granularity, and with
    // SILENCE_SEC at 0.25s an 85ms block was too coarse to find the real gap
    // between phrases.
    proc = audioCtx.createScriptProcessor(2048, 1, 1);
    muteNode = audioCtx.createGain();
    muteNode.gain.value = 0;               // keep the graph running silently
    mixer.connect(proc);
    proc.connect(muteNode);
    muteNode.connect(audioCtx.destination);
    proc.onaudioprocess = onAudio;

    // If the user ends the screen share, or a device is unplugged, stop cleanly.
    for (const s of streams)
      s.getTracks().forEach((t) => { t.onended = onTrackEnded; });

    if (mode !== "system") await populateMics();   // labels available now
  }

  function onTrackEnded() {
    if (!running) return;
    stopCapture();
    setUIRunning(false);
    setBadge(false, "Idle");
    setStatus("stopped", "muted");
    setVoice("idle", "muted");
    showNotice("Audio capture ended — screen sharing was stopped or the device " +
               "was removed. Press Start Translation to go again.", "info");
  }

  function stopCapture() {
    running = false;
    try { if (proc) { proc.onaudioprocess = null; proc.disconnect(); } } catch (_) {}
    try { if (muteNode) muteNode.disconnect(); } catch (_) {}
    try { if (mixer) mixer.disconnect(); } catch (_) {}
    for (const n of srcNodes) { try { n.disconnect(); } catch (_) {} }
    for (const s of streams)
      s.getTracks().forEach((t) => { t.onended = null; try { t.stop(); } catch (_) {} });
    try { if (audioCtx) audioCtx.close(); } catch (_) {}
    proc = muteNode = mixer = audioCtx = null;
    srcNodes = []; streams = [];
    vad.inSpeech = false; vad.buffer = [];
    vad.speechSamples = vad.silenceSamples = vad.totalSamples = 0;
    vad.interimAt = 0; vad.part = 0;
    noiseFloor = 0;
    resetMeter();
  }

  // ---- the audio callback: meter + VAD ------------------------------------
  function onAudio(e) {
    if (!running) return;
    const input = e.inputBuffer.getChannelData(0);
    const rms = computeRMS(input);

    // While speaking the translation, keep the meter alive but discard audio
    // (so we don't re-capture our own output = a feedback loop). This matters
    // most in system-audio mode, where our TTS *is* part of what we capture.
    // A short cooldown covers the speaker/room tail after playback ends.
    if (speaking || Date.now() < quietUntil) { updateMeter(rms, false); return; }

    // Adapt to the source. System audio carries music, effects and room tone
    // under the dialogue; against a fixed threshold that background never drops
    // below the line, so an utterance never ends naturally and every chunk is a
    // force-cut mid-sentence. Tracking the quiet baseline and requiring speech
    // to stand NOISE_MULT times above it restores real phrase boundaries.
    //
    // The floor falls quickly (a genuine quiet moment is information) and rises
    // slowly (so one long sentence can't drag the baseline up to its own level).
    // TH remains the absolute minimum, so a microphone in a quiet room - where
    // the baseline is near zero - behaves exactly as it did before.
    if (noiseFloor === 0)
      // A loud first block is more likely speech than background. Calibrate
      // from the absolute floor so it cannot suppress the opening word.
      noiseFloor = Math.min(rms, TH);
    noiseFloor = rms < noiseFloor ? noiseFloor * 0.85 + rms * 0.15
                                  : noiseFloor * 0.995 + rms * 0.005;
    const threshold = Math.max(TH, noiseFloor * NOISE_MULT);

    const loud = rms >= threshold;
    updateMeter(rms, loud);

    const n = input.length;
    if (loud) {
      if (!vad.inSpeech) startUtterance();
      vad.speechSamples += n;
      vad.silenceSamples = 0;
      vad.buffer.push(new Float32Array(input));   // copy: input is reused
      vad.totalSamples += n;
    } else if (vad.inSpeech) {
      vad.buffer.push(new Float32Array(input));   // keep trailing silence
      vad.totalSamples += n;
      vad.silenceSamples += n;
      if (vad.silenceSamples / RATE >= SILENCE_SEC &&
          vad.speechSamples / RATE >= MIN_SPEECH_SEC) {
        finalizeUtterance(false);
        return;
      }
    }

    if (!vad.inSpeech) return;

    // Force-cut a long unbroken stretch so we don't wait for a pause.
    if (vad.totalSamples / RATE >= MAX_UTTER_SEC) { finalizeUtterance(true); return; }

    // Provisional caption: translate what we have so far, mid-sentence, so the
    // screen keeps up with a video instead of staying blank for the whole chunk.
    if (interimOn && (vad.totalSamples - vad.interimAt) / RATE >= INTERIM_SEC) {
      vad.interimAt = vad.totalSamples;
      vad.part += 1;
      sendChunk(flatten(vad.buffer, vad.totalSamples), vad.uid, vad.part, true);
    }
  }

  function startUtterance() {
    vad.inSpeech = true;
    vad.speechSamples = 0; vad.silenceSamples = 0; vad.totalSamples = 0;
    vad.buffer = [];
    vad.uid = ++uidCounter;
    vad.part = 0;
    vad.interimAt = 0;
    setBadge(true, "Hearing speech…");
  }

  /** Concatenate the block list into one Float32Array. */
  function flatten(blocks, total) {
    const out = new Float32Array(total);
    let off = 0;
    for (const b of blocks) { out.set(b, off); off += b.length; }
    return out;
  }

  /**
   * End the current chunk and send it as the authoritative result.
   * `forced` means we hit MAX_UTTER_SEC mid-speech. Rather than slicing at that
   * exact instant (which usually lands mid-word), we cut at the quietest moment
   * in the preceding second and carry EVERYTHING after the cut into the next
   * chunk - so no audio is lost, nothing is duplicated, and both chunks contain
   * whole words. We also stay "in speech" so the meter/badge don't flicker.
   */
  function finalizeUtterance(forced) {
    const speechSec = vad.speechSamples / RATE;
    const total = vad.totalSamples;
    const uid = vad.uid;
    let samples = total ? flatten(vad.buffer, total) : null;

    if (forced) {
      let carry = new Float32Array(0);
      if (samples) {
        const cut = quietestPoint(samples, Math.round(CUT_SEARCH_SEC * RATE));
        carry = samples.slice(cut);
        samples = samples.slice(0, cut);
      }
      vad.buffer = carry.length ? [carry] : [];
      vad.totalSamples = carry.length;
      vad.speechSamples = carry.length;   // still mid-speech; don't drop the tail
      vad.silenceSamples = 0;
      vad.uid = ++uidCounter;
      vad.part = 0;
      vad.interimAt = 0;
    } else {
      vad.inSpeech = false;
      vad.buffer = [];
      vad.speechSamples = vad.silenceSamples = vad.totalSamples = 0;
      if (running) setBadge(true, "Listening…");
    }

    if (!samples || !samples.length || speechSec < MIN_SPEECH_SEC) return;
    sendChunk(samples, uid, 0, false);
  }

  // ---- talk to the server -------------------------------------------------
  /**
   * Send one chunk of speech. Requests overlap (the server handles them
   * concurrently), so every reply is checked against what is already on screen
   * before it is allowed to draw.
   */
  async function sendChunk(samples, uid, part, interim) {
    const pcm = floatTo16BitPCM(normalize(samples));
    const q = `source=${enc(sourceSel.value)}&target=${enc(targetSel.value)}` +
              `&rate=${RATE}&uid=${uid}&part=${part}&interim=${interim ? 1 : 0}`;
    if (!interim) setStatus("translating…", "accent");

    let data;
    try {
      const res = await fetch(`/api/translate?${q}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: pcm,
      });
      data = await res.json().catch(() => ({ error: "bad server response" }));
    } catch (err) {
      if (!interim) {
        setStatus("network error", "stop");
        showNotice("Network error: " + err.message, "err");
      }
      return;
    }

    if (data.error) {
      // A failing interim is not worth bothering the user about - but if they
      // keep failing (rate limit), stop sending them and keep finals working.
      if (interim) {
        if (++interimFails >= 3) {
          interimOn = false;
          showNotice("Live captions turned off — the free speech service is " +
                     "rate-limiting the extra requests. Translations still " +
                     "appear at the end of each phrase.", "warn");
        }
        return;
      }
      if (data.original) showInterim(data.original, "");
      if (data.detected) setDetected(data.detected);
      setStatus("error", "stop");
      showNotice(data.error, "err");
      return;
    }
    if (interim) interimFails = 0;

    // How sure the recognizer was. Shown next to the latency so a suspicious
    // line can be told apart from a confident one - useful when checking the
    // translation against what you can hear.
    const conf = typeof data.confidence === "number"
      ? ` · ${Math.round(data.confidence * 100)}%` : "";

    // ---- ordering guards ----
    if (interim) {
      // Superseded once this utterance's final (or any later one) has landed.
      if (uid <= lastFinalUid) return;
      const key = uid * 1000 + part;
      if (key <= lastInterimKey) return;      // an older interim arrived late
      lastInterimKey = key;
    } else {
      if (uid < lastFinalUid) return;         // a stale final arrived late
      lastFinalUid = uid;
    }

    if (data.empty) {
      // `rejected` means the recognizer DID return words, but below the
      // confidence bar - almost always music or noise. Say so, rather than
      // silently doing nothing or printing the invented words.
      if (!interim)
        setStatus(data.rejected ? `ignored (noise)${conf}` : "listening",
                  data.rejected ? "muted" : "start");
      return;
    }

    if (data.detected) setDetected(data.detected);
    hideNotice();

    if (interim) {
      showInterim(data.original, data.translated);
      setStatus(`live · ${data.ms} ms${conf}`, "accent");
      return;
    }

    commitLine(data.original, data.translated);
    setStatus(`translated ✓ · ${data.ms} ms${conf}`, "start");
    if (speakChk.checked) speak(data.translated);
  }

  async function speak(text) {
    if (!text) return;
    speaking = true;
    setVoice("speaking…", "accent");
    // Anything already buffered was captured before/while we started talking.
    const done = (msg) => {
      speaking = false;
      quietUntil = Date.now() + 300;   // ignore the speaker tail
      setVoice(msg, "muted");
    };
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, target: targetSel.value }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        done(e.error || "voice unavailable");
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      ttsPlayer.src = url;
      ttsPlayer.onended = () => { URL.revokeObjectURL(url); done("idle"); };
      ttsPlayer.onerror = () => done("playback failed");
      await ttsPlayer.play();
    } catch (err) {
      done("voice error");
    }
  }

  // ---- UI running state ---------------------------------------------------
  function setUIRunning(on) {
    running = on;
    startBtn.disabled = on;
    stopBtn.disabled = !on;
    sourceSel.disabled = on;
    targetSel.disabled = on;
    micSel.disabled = on;
    audioSrcSel.disabled = on;
    refreshBtn.disabled = on;
  }

  /** Show the hint for the chosen source, and hide the mic picker when unused. */
  function syncSourceMode() {
    const mode = audioSrcSel.value;
    let hint = HINTS[mode] || "";
    if (sourceSel.value === CFG.autoLabel)
      hint += (hint ? " " : "") +
        "Auto source detection is best-effort; choose the spoken language " +
        "explicitly for the most accurate result.";
    srcHint.textContent = hint;
    micField.hidden = (mode === "system");
    refreshBtn.hidden = (mode === "system");

    // Capture is muted while we speak (otherwise we translate our own voice).
    // With system audio that means missing part of what's playing, so default
    // to subtitles-only when following a video. The user can switch it back.
    if (mode !== "mic" && speakChk.checked) {
      speakChk.checked = false;
      speakNote.hidden = false;
    } else if (mode === "mic") {
      speakNote.hidden = true;
    }
  }

  function captureErrorMessage(err, mode) {
    const name = err && err.name;
    const msg = err && err.message;
    const system = (mode === "system" || mode === "both");

    if (msg === "no-mic-api")
      return "This browser can't access the microphone here. Open the site via http://127.0.0.1 (localhost) or HTTPS.";
    if (msg === "no-display-api")
      return "This browser can't capture system audio. Use Chrome or Edge on the desktop, " +
             "or switch Audio source to Microphone.";
    if (msg === "no-system-audio")
      return "You shared a screen, but not its audio. Press Start Translation again and " +
             "tick “Also share system audio” (bottom-left of the share dialog) — or share a " +
             "single browser tab, whose audio is always included.";
    if (name === "NotAllowedError" || name === "SecurityError")
      return system
        ? "Screen sharing was cancelled or blocked, so there's no system audio to translate. " +
          "Press Start Translation and choose a screen or tab to share."
        : "Microphone permission was denied. Allow mic access in your browser and try again.";
    if (name === "NotFoundError" || name === "OverconstrainedError")
      return "No microphone was found. Plug one in, click Refresh, and try again.";
    if (name === "NotReadableError")
      return "The microphone is in use by another app. Close it and try again.";
    if (name === "InvalidStateError")
      return "The browser wouldn't open the share dialog. Click Start Translation again.";
    return "Could not start audio capture: " + (msg || name || err);
  }

  // ---- wire up ------------------------------------------------------------
  startBtn.addEventListener("click", async () => {
    const mode = audioSrcSel.value;
    hideNotice();
    noiseFloor = 0;
    interimOn = INTERIM_SEC > 0;
    interimFails = 0;
    setUIRunning(true);
    if (mode !== "mic") setStatus("waiting for share…", "accent");
    try {
      await startCapture();
      running = true;
      setBadge(true, "Listening…");
      setStatus("listening", "start");
      setVoice("idle", "muted");
    } catch (err) {
      stopCapture();                       // release anything already opened
      setUIRunning(false);
      setBadge(false, "Idle");
      setStatus(mode === "mic" ? "mic error" : "capture error", "stop");
      showNotice(captureErrorMessage(err, mode), "err");
    }
  });

  stopBtn.addEventListener("click", () => {
    stopCapture();
    setUIRunning(false);
    setBadge(false, "Idle");
    setStatus("idle", "muted");
    setVoice("idle", "muted");
  });

  clearBtn.addEventListener("click", async () => {
    clearCaptions();
    setDetected("—");
    hideNotice();
    try { await fetch("/api/clear", { method: "POST" }); } catch (_) {}
  });

  refreshBtn.addEventListener("click", populateMics);
  audioSrcSel.addEventListener("change", syncSourceMode);
  sourceSel.addEventListener("change", syncSourceMode);

  // ---- init ---------------------------------------------------------------
  function init() {
    buildMeter();
    fillSelect(sourceSel, CFG.sourceLanguages, CFG.defaultSource);
    fillSelect(targetSel, CFG.targetLanguages, CFG.defaultTarget);
    speakChk.checked = CFG.speakDefault !== false;
    clearCaptions();
    setStatus("idle", "muted"); setVoice("idle", "muted");
    populateMics();

    // Only Chromium browsers can share system audio - say so instead of
    // offering an option that silently won't work.
    const canShare = !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
    if (!canShare)
      for (const o of audioSrcSel.options)
        if (o.value !== "mic") { o.disabled = true; o.textContent += " (needs Chrome/Edge)"; }
    syncSourceMode();

    const localish = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
    if (!window.isSecureContext && !localish) {
      showNotice("Microphone needs HTTPS or localhost. If this is a LAN address, " +
                 "open it as http://127.0.0.1:" + (location.port || "5000") + " on the same PC.", "warn");
    }
  }

  init();
})();
