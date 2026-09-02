"""
gui.py
------
The Tkinter user interface for the AI Voice Translator.

Design notes for the student:
    * Tkinter is NOT thread-safe. Only the main thread may touch widgets.
      Background workers (audio / STT / translation / TTS) therefore never call
      widget methods directly. Instead they call the public helper methods here
      (set_subtitle, set_level, ...), which just drop a small message onto an
      internal queue.Queue. The GUI drains that queue on the main thread every
      50 ms using root.after(). This is the standard, safe way to update a
      Tkinter GUI from other threads and is what keeps the window responsive.

    * This file has NO dependency on any audio/AI library, so it can run on its
      own before those are installed. main.py injects the real callbacks later.

Layout (top -> bottom):
    ┌ header ─ title + live status badge (pulsing dot) ────────────┐
    │ controls card ─ From / To / Audio input / Refresh            │
    │ input-level meter ─ segmented bar, reacts live as you speak  │
    │ actions ─ Start / Stop / Clear + Speak toggle                │
    │ status pills ─ Detected / Translation / Voice                │
    │ "You said" card ─ the recognised original speech             │
    │ "Translation" card ─ the BIG live subtitle (fills the space) │
    └──────────────────────────────────────────────────────────────┘
"""

import queue
import tkinter as tk
from tkinter import ttk

import config

PAD = 18          # outer padding used consistently across the window
NUM_SEGMENTS = 28  # bars in the live input-level meter


class TranslatorGUI:
    def __init__(self, root,
                 on_start=None, on_stop=None, on_clear=None,
                 list_input_devices=None):
        """
        root                : the tk.Tk() root window
        on_start(settings)  : called when Start is pressed (settings = dict)
        on_stop()           : called when Stop is pressed
        on_clear()          : called when Clear Subtitles is pressed
        list_input_devices(): returns a list of {"label": str, "id": any}
        """
        self.root = root
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_clear = on_clear
        self.list_input_devices = list_input_devices or (
            lambda: [{"label": "Default input", "id": None}]
        )

        # Messages from worker threads are queued here and applied on the main
        # thread by _poll_queue(). Each message is a (method_name, args) tuple.
        self._ui_queue = queue.Queue()

        self._devices = []              # cached device list [{label,id}, ...]
        self._running = False
        self._indicator_active = False  # drives the pulsing live badge
        self._pulse_on = False
        self._level_segments = []       # canvas ids for the meter bars

        self._build_style()
        self._build_widgets()
        self._refresh_devices()

        # Start the periodic loops on the main thread.
        self.root.after(50, self._poll_queue)
        self.root.after(500, self._pulse)

    # ------------------------------------------------------------------ style
    def _build_style(self):
        c = config.COLORS
        self.root.title(config.APP_TITLE)
        self.root.geometry(config.WINDOW_SIZE)
        self.root.minsize(820, 640)
        self.root.configure(bg=c["bg"])

        style = ttk.Style()
        # 'clam' gives a consistent, themeable look across Windows.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Comboboxes: flat, dark, with an accent arrow. style.map keeps the text
        # readable in the readonly/disabled states (clam otherwise washes it out).
        style.configure("TCombobox",
                        fieldbackground=c["card"], background=c["card"],
                        foreground=c["text"], arrowcolor=c["accent"],
                        bordercolor=c["border"], lightcolor=c["border"],
                        darkcolor=c["border"], relief="flat", padding=6)
        style.map("TCombobox",
                  fieldbackground=[("readonly", c["card"]),
                                   ("disabled", c["panel"])],
                  foreground=[("disabled", c["muted"])],
                  arrowcolor=[("disabled", c["muted"])],
                  selectbackground=[("readonly", c["card"])],
                  selectforeground=[("readonly", c["text"])])

        # The dropdown list that pops open from a combobox.
        self.root.option_add("*TCombobox*Listbox.background", c["card"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["accent2"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", c["bg"])
        self.root.option_add("*TCombobox*Listbox.font", config.UI_FONT)

    # ----------------------------------------------------------- small helpers
    def _card(self, parent, bg=None):
        """A panel with a subtle 1px border. Returns the inner frame to fill."""
        c = config.COLORS
        outer = tk.Frame(parent, bg=c["border"])
        inner = tk.Frame(outer, bg=bg or c["card"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        outer._inner = inner
        return outer

    def _make_button(self, parent, text, cmd, base, fg, hover, font=None,
                     **kw):
        """Flat button with a hover effect (skipped while disabled)."""
        b = tk.Button(parent, text=text, command=cmd, bg=base, fg=fg,
                      activebackground=hover, activeforeground=fg,
                      disabledforeground=config.COLORS["muted"],
                      relief="flat", bd=0, cursor="hand2",
                      font=font or config.UI_FONT, **kw)
        b._base, b._hover = base, hover
        b.bind("<Enter>", self._on_btn_enter)
        b.bind("<Leave>", self._on_btn_leave)
        return b

    @staticmethod
    def _on_btn_enter(event):
        w = event.widget
        if str(w["state"]) != "disabled":
            w.configure(bg=w._hover)

    @staticmethod
    def _on_btn_leave(event):
        w = event.widget
        if str(w["state"]) != "disabled":
            w.configure(bg=w._base)

    def _pill(self, parent, title):
        """A small titled status card. Returns the value label to update."""
        c = config.COLORS
        card = self._card(parent)
        inner = card._inner
        tk.Label(inner, text=title.upper(), bg=c["card"], fg=c["muted"],
                 font=config.LABEL_FONT).pack(anchor="w", padx=12, pady=(8, 0))
        value = tk.Label(inner, text="—", bg=c["card"], fg=c["text"],
                         font=config.STATUS_FONT, anchor="w")
        value.pack(anchor="w", fill="x", padx=12, pady=(0, 9))
        return card, value

    # ----------------------------------------------------------------- widgets
    def _build_widgets(self):
        c = config.COLORS

        # ============================ HEADER ==============================
        header = tk.Frame(self.root, bg=c["bg"])
        header.pack(fill="x", padx=PAD, pady=(PAD, 8))

        title_box = tk.Frame(header, bg=c["bg"])
        title_box.pack(side="left")
        tk.Label(title_box, text="🎙  AI Voice Translator", bg=c["bg"],
                 fg=c["text"], font=config.TITLE_FONT).pack(anchor="w")
        tk.Label(title_box, text="Live speech translation", bg=c["bg"],
                 fg=c["muted"], font=config.LABEL_FONT).pack(anchor="w")

        # Live status badge (bordered pill with a pulsing dot).
        badge = self._card(header)
        badge.pack(side="right")
        bi = badge._inner
        self.indicator_dot = tk.Label(bi, text="●", bg=c["card"], fg=c["idle"],
                                      font=config.BADGE_FONT)
        self.indicator_dot.pack(side="left", padx=(12, 6), pady=6)
        self.indicator = tk.Label(bi, text="Idle", bg=c["card"], fg=c["muted"],
                                  font=config.BADGE_FONT)
        self.indicator.pack(side="left", padx=(0, 12), pady=6)

        # =========================== CONTROLS =============================
        controls_card = self._card(self.root)
        controls_card.pack(fill="x", padx=PAD, pady=6)
        controls = controls_card._inner
        controls.configure(padx=14, pady=14)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        # Row 0: From -> To
        tk.Label(controls, text="From", bg=c["card"], fg=c["muted"],
                 font=config.LABEL_FONT).grid(row=0, column=0, padx=(0, 8),
                                              sticky="w")
        self.source_var = tk.StringVar(value=config.DEFAULT_SOURCE)
        self.source_combo = ttk.Combobox(
            controls, textvariable=self.source_var, state="readonly",
            values=config.source_language_names(), font=config.UI_FONT)
        self.source_combo.grid(row=0, column=1, padx=(0, 12), sticky="we")

        tk.Label(controls, text="→", bg=c["card"], fg=c["accent"],
                 font=("Segoe UI", 14, "bold")).grid(row=0, column=2, padx=6)

        tk.Label(controls, text="To", bg=c["card"], fg=c["muted"],
                 font=config.LABEL_FONT).grid(row=0, column=3, padx=(6, 8),
                                              sticky="w")
        self.target_var = tk.StringVar(value=config.DEFAULT_TARGET)
        self.target_combo = ttk.Combobox(
            controls, textvariable=self.target_var, state="readonly",
            values=config.target_language_names(), font=config.UI_FONT)
        self.target_combo.grid(row=0, column=4, sticky="we")

        # Row 1: Audio input device + Refresh
        tk.Label(controls, text="Audio input", bg=c["card"], fg=c["muted"],
                 font=config.LABEL_FONT).grid(row=1, column=0, padx=(0, 8),
                                              pady=(12, 0), sticky="w")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            controls, textvariable=self.device_var, state="readonly",
            font=config.UI_FONT)
        self.device_combo.grid(row=1, column=1, columnspan=3, pady=(12, 0),
                               sticky="we")
        self.refresh_btn = self._make_button(
            controls, "⟳  Refresh", self._refresh_devices,
            base=c["card"], fg=c["text"], hover=c["border"],
            padx=12, pady=6, highlightbackground=c["border"],
            highlightthickness=1)
        self.refresh_btn.grid(row=1, column=4, padx=(12, 0), pady=(12, 0),
                              sticky="we")

        # ======================== INPUT LEVEL METER =======================
        meter_row = tk.Frame(self.root, bg=c["bg"])
        meter_row.pack(fill="x", padx=PAD, pady=(8, 2))
        tk.Label(meter_row, text="INPUT LEVEL", bg=c["bg"], fg=c["muted"],
                 font=config.LABEL_FONT).pack(side="left", padx=(2, 10))

        self._meter_h = 18
        self.level_canvas = tk.Canvas(meter_row, height=self._meter_h,
                                      bg=c["bg"], highlightthickness=0, bd=0)
        self.level_canvas.pack(side="left", fill="x", expand=True)
        # Bars are drawn on first configure so they span the real width.
        self.level_canvas.bind("<Configure>", self._draw_meter)

        # =========================== ACTIONS ==============================
        actions = tk.Frame(self.root, bg=c["bg"])
        actions.pack(fill="x", padx=PAD, pady=(10, 6))

        self.start_btn = self._make_button(
            actions, "▶  Start Translation", self._handle_start,
            base=c["start"], fg="#0c1f14", hover="#5cf08c",
            font=("Segoe UI Semibold", 11), padx=16, pady=9)
        self.start_btn.pack(side="left")

        self.stop_btn = self._make_button(
            actions, "■  Stop", self._handle_stop,
            base=c["card"], fg=c["muted"], hover=c["border"],
            font=("Segoe UI Semibold", 11), padx=16, pady=9,
            state="disabled")
        self.stop_btn.pack(side="left", padx=(10, 0))

        self.clear_btn = self._make_button(
            actions, "🧹  Clear", self._handle_clear,
            base=c["card"], fg=c["text"], hover=c["border"],
            padx=14, pady=9, highlightbackground=c["border"],
            highlightthickness=1)
        self.clear_btn.pack(side="left", padx=(10, 0))

        self.speak_var = tk.BooleanVar(value=config.SPEAK_ENABLED_BY_DEFAULT)
        self.speak_check = tk.Checkbutton(
            actions, text="🔊  Speak translation", variable=self.speak_var,
            bg=c["bg"], fg=c["text"], selectcolor=c["card"],
            activebackground=c["bg"], activeforeground=c["text"],
            font=config.UI_FONT, bd=0, highlightthickness=0, cursor="hand2")
        self.speak_check.pack(side="right")

        # ========================= STATUS PILLS ===========================
        pills = tk.Frame(self.root, bg=c["bg"])
        pills.pack(fill="x", padx=PAD, pady=(6, 6))
        for i in range(3):
            pills.columnconfigure(i, weight=1, uniform="pill")

        p1, self.lang_value = self._pill(pills, "Detected language")
        p1.grid(row=0, column=0, sticky="we", padx=(0, 6))
        p2, self.trans_value = self._pill(pills, "Translation")
        p2.grid(row=0, column=1, sticky="we", padx=6)
        p3, self.tts_value = self._pill(pills, "Voice output")
        p3.grid(row=0, column=2, sticky="we", padx=(6, 0))
        self._apply_trans_status("idle", c["muted"])
        self._apply_tts_status("idle", c["muted"])

        # ===================== ORIGINAL (You said) ========================
        said_card = self._card(self.root)
        said_card.pack(fill="x", padx=PAD, pady=(6, 6))
        said = said_card._inner
        tk.Label(said, text="YOU SAID", bg=c["card"], fg=c["muted"],
                 font=config.LABEL_FONT).pack(anchor="w", padx=14, pady=(10, 2))
        self.original_text = tk.Text(
            said, height=3, wrap="word", bg=c["card"], fg=c["text"],
            font=config.ORIGINAL_FONT, relief="flat", padx=14, pady=(0),
            insertbackground=c["text"], highlightthickness=0, bd=0)
        self.original_text.pack(fill="x", padx=2, pady=(0, 10))
        self.original_text.tag_configure("left", justify="left")
        self.original_text.configure(state="disabled")

        # ================= TRANSLATION (big subtitle) =====================
        sub_card = self._card(self.root, bg=c["panel"])
        sub_card.pack(fill="both", expand=True, padx=PAD, pady=(6, PAD))
        sub = sub_card._inner
        tk.Label(sub, text="TRANSLATION", bg=c["panel"], fg=c["accent"],
                 font=config.LABEL_FONT).pack(anchor="w", padx=16, pady=(12, 4))
        self.subtitle_text = tk.Text(
            sub, wrap="word", bg=c["panel"], fg=c["accent"],
            font=config.SUBTITLE_FONT, relief="flat", padx=18, pady=8,
            insertbackground=c["text"], highlightthickness=0, bd=0)
        self.subtitle_text.pack(fill="both", expand=True, padx=4, pady=(0, 14))
        self.subtitle_text.tag_configure("center", justify="center")
        self.subtitle_text.configure(state="disabled")

        # Show friendly placeholders until real content arrives.
        self._apply_original("")
        self._apply_subtitle("")

    # ------------------------------------------------------------ level meter
    def _draw_meter(self, event=None):
        """(Re)build the meter bars to fill the current canvas width."""
        c = config.COLORS
        cv = self.level_canvas
        cv.delete("all")
        self._level_segments = []
        w = cv.winfo_width() or 360
        h = self._meter_h
        gap = 3
        seg_w = max(2, (w - (NUM_SEGMENTS - 1) * gap) / NUM_SEGMENTS)
        x = 0
        for _ in range(NUM_SEGMENTS):
            seg = cv.create_rectangle(x, 2, x + seg_w, h - 2,
                                      fill=c["level_bg"], width=0)
            self._level_segments.append(seg)
            x += seg_w + gap

    @staticmethod
    def _seg_color(frac, is_speech):
        c = config.COLORS
        if not is_speech:
            return c["idle"]
        if frac < 0.6:
            return c["start"]
        if frac < 0.85:
            return c["accent"]
        return c["stop"]

    def _apply_level(self, rms, is_speech):
        if not self._level_segments:
            return
        c = config.COLORS
        # rms is 0..1 but real speech sits low (~0.02-0.15); sqrt + gain makes
        # it perceptually visible without pinning to the top.
        level = min(1.0, (max(0.0, rms) ** 0.5) * 2.4)
        lit = int(round(level * NUM_SEGMENTS))
        n = max(1, NUM_SEGMENTS - 1)
        for i, seg in enumerate(self._level_segments):
            color = self._seg_color(i / n, is_speech) if i < lit \
                else c["level_bg"]
            self.level_canvas.itemconfigure(seg, fill=color)

    # ------------------------------------------------------- device handling
    def _refresh_devices(self):
        """Query the injected device provider and repopulate the dropdown."""
        try:
            self._devices = list(self.list_input_devices())
        except Exception as exc:  # provider might fail if audio lib missing
            self._devices = [{"label": f"(device query failed: {exc})",
                              "id": None}]
        labels = [d["label"] for d in self._devices] or ["(no input devices)"]
        self.device_combo.configure(values=labels)
        # keep current selection if still present, else pick the first
        if self.device_var.get() not in labels:
            self.device_var.set(labels[0])

    def _selected_device_id(self):
        label = self.device_var.get()
        for d in self._devices:
            if d["label"] == label:
                return d["id"]
        return None

    # ------------------------------------------------------- button handlers
    def get_settings(self):
        """Snapshot of the current user selections, passed to on_start."""
        return {
            "source_name": self.source_var.get(),
            "target_name": self.target_var.get(),
            "device_id": self._selected_device_id(),
            "device_label": self.device_var.get(),
            "speak": bool(self.speak_var.get()),
        }

    def _handle_start(self):
        settings = self.get_settings()
        self.set_running_ui(True)
        if self.on_start:
            self.on_start(settings)

    def _handle_stop(self):
        if self.on_stop:
            self.on_stop()
        self.set_running_ui(False)

    def _handle_clear(self):
        self.set_original("")
        self.set_subtitle("")
        self.set_detected_language("—")
        if self.on_clear:
            self.on_clear()

    def set_running_ui(self, running):
        """Enable/disable controls to reflect running state (main thread)."""
        c = config.COLORS
        self._running = running
        if running:
            self.start_btn.configure(state="disabled", bg=c["card"])
            self.stop_btn.configure(state="normal", bg=c["stop"], fg="#2a0d0d")
            for w in (self.source_combo, self.target_combo,
                      self.device_combo):
                w.configure(state="disabled")
            self.refresh_btn.configure(state="disabled")
            self.set_input_indicator(True, "Listening…")
            self.set_translation_status("running", c["start"])
        else:
            self.start_btn.configure(state="normal", bg=c["start"])
            self.stop_btn.configure(state="disabled", bg=c["card"],
                                    fg=c["muted"])
            self.source_combo.configure(state="readonly")
            self.target_combo.configure(state="readonly")
            self.device_combo.configure(state="readonly")
            self.refresh_btn.configure(state="normal")
            self.set_input_indicator(False, "Idle")
            self.set_translation_status("idle", c["muted"])
            self.set_tts_status("idle", c["muted"])
            self._apply_level(0.0, False)

    # --------------------------------------------------------------- helpers
    def _set_text_widget(self, widget, text, tag="left"):
        """Replace the contents of a read-only Text widget (with a justify tag)."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text, tag)
        widget.configure(state="disabled")

    # -------------------------------------------------------------------------
    # Public, THREAD-SAFE update methods.
    # Worker threads call these; they just enqueue a request that _poll_queue
    # applies on the main thread. Safe to call from any thread.
    # -------------------------------------------------------------------------
    def set_original(self, text):
        self._ui_queue.put(("_apply_original", (text,)))

    def set_subtitle(self, text):
        self._ui_queue.put(("_apply_subtitle", (text,)))

    def set_detected_language(self, name):
        self._ui_queue.put(("_apply_detected_language", (name,)))

    def set_translation_status(self, text, color=None):
        self._ui_queue.put(("_apply_trans_status", (text, color)))

    def set_tts_status(self, text, color=None):
        self._ui_queue.put(("_apply_tts_status", (text, color)))

    def set_input_indicator(self, active, label=None):
        self._ui_queue.put(("_apply_indicator", (active, label)))

    def set_level(self, rms, is_speech):
        """Feed the live input-level meter (rms in 0..1)."""
        self._ui_queue.put(("_apply_level", (rms, is_speech)))

    def notify_stopped(self):
        """Worker can call this when the pipeline has fully stopped."""
        self._ui_queue.put(("set_running_ui", (False,)))

    # ---- the actual widget mutations (run on main thread only) ------------
    def _apply_original(self, text):
        c = config.COLORS
        if text:
            self.original_text.configure(fg=c["text"])
            self._set_text_widget(self.original_text, text, "left")
        else:
            self.original_text.configure(fg=c["muted"])
            self._set_text_widget(self.original_text,
                                  "Waiting for speech…", "left")

    def _apply_subtitle(self, text):
        c = config.COLORS
        if text:
            self.subtitle_text.configure(fg=c["accent"])
            self._set_text_widget(self.subtitle_text, text, "center")
        else:
            self.subtitle_text.configure(fg=c["muted"])
            self._set_text_widget(self.subtitle_text,
                                  "Translation will appear here…", "center")

    def _apply_detected_language(self, name):
        self.lang_value.configure(text=name or "—")

    def _apply_trans_status(self, text, color):
        self.trans_value.configure(text=text,
                                   fg=color or config.COLORS["text"])

    def _apply_tts_status(self, text, color):
        self.tts_value.configure(text=text,
                                 fg=color or config.COLORS["text"])

    def _apply_indicator(self, active, label):
        c = config.COLORS
        self._indicator_active = active
        text = label or ("Listening…" if active else "Idle")
        self.indicator.configure(text=text,
                                 fg=c["start"] if active else c["muted"])
        self.indicator_dot.configure(fg=c["start"] if active else c["idle"])

    # ------------------------------------------------------- main-thread loops
    def _pulse(self):
        """Gently pulse the live badge dot while active (main thread)."""
        c = config.COLORS
        if self._indicator_active:
            self._pulse_on = not self._pulse_on
            self.indicator_dot.configure(fg=c["start"] if self._pulse_on
                                         else c["accent"])
        self.root.after(500, self._pulse)

    def _poll_queue(self):
        """Drain all pending UI messages, then reschedule. Main thread only."""
        try:
            while True:
                method_name, args = self._ui_queue.get_nowait()
                try:
                    getattr(self, method_name)(*args)
                except Exception:
                    # Never let one bad update kill the whole UI loop.
                    pass
        except queue.Empty:
            pass
        # Reschedule the next drain (keeps the GUI responsive).
        self.root.after(50, self._poll_queue)
