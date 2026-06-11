"""
═══════════════════════════════════════════════════════════════════
 SWAHILI VOICE MASTER — PIPELINE v2.0
 Professional AI movie dubbing: English/Korean/any → Kiswahili
═══════════════════════════════════════════════════════════════════

 FIXES vs v1 (kwa nini v1 ilitoa output mbaya):
 1. EMOTION FIX: per-segment prosody cloning. Kila line ya TTS
    ina-clone sauti + hisia kutoka kwenye SEGMENT YENYEWE ya
    original (kutoka separated vocals), sio sample moja ya 20s.
    Mhusika akipiga kelele kwenye original → TTS inapiga kelele.
 2. PRONUNCIATION FIX: XTTS-v2 haina Swahili. v1 ilitumia "en"
    (vokali mbovu). v2 inatumia "es" token — Spanish ina vokali
    safi a-e-i-o-u kama Kiswahili + rolled R. Tofauti ni kubwa.
 3. HALLUCINATION FIX: faster-whisper large-v3 + Silero VAD +
    no_speech_prob / avg_logprob / repetition filters. Hakuna
    tena "maneno ya uongo" kwenye music/silence.
 4. TIMING FIX: kila TTS segment ina-time-stretch (atempo) kufit
    duration ya original ± tolerance. Hakuna overlap, hakuna drift.
 5. CLEAN REFERENCE FIX: voice references zinakatwa kutoka
    SEPARATED VOCALS (sio original movie) — hakuna music/SFX
    ndani ya voice clone.
 6. MIX FIX: dialogue bus loudness-normalized (-16 LUFS),
    background full level + sidechain ducking kidogo wakati wa
    dialogue, master loudnorm. Background sasa inasikika vizuri.

 BONUS:
 - Checkpoint/resume kila step (Kaggle 12h session-proof)
 - Cross-platform (Kaggle Linux + Windows local)
 - Batch translation yenye scene context (faster + coherent)

 USAGE (Kaggle):
   python pipeline_v2.py --input /kaggle/input/movies/film.mp4 \
                         --output /kaggle/working/out
 USAGE (Windows):
   python pipeline_v2.py --input "C:\\movies\\film.mp4" --output "C:\\out"
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import shutil
import argparse
import subprocess
from pathlib import Path

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
CONFIG = {
    # Whisper
    "whisper_model": "large-v3",          # large-v3 >> medium kwa accuracy
    "whisper_compute": "float16",         # "int8" kama GPU memory ndogo
    "vad_min_silence_ms": 400,

    # Hallucination filters
    "max_no_speech_prob": 0.55,
    "min_avg_logprob": -1.0,
    "max_compression_ratio": 2.4,

    # Demucs
    "demucs_model": "htdemucs_ft",        # best quality; "htdemucs" = 4x faster

    # Translation (Groq — free tier)
    "groq_model": "llama-3.3-70b-versatile",
    "translate_batch_size": 15,           # segments per API call (scene context)

    # TTS
    "xtts_lang": "es",                    # "es" inasoma Kiswahili bora kuliko "en"
    "min_ref_seconds": 3.0,               # XTTS inahitaji ref >= ~3s
    "max_ref_seconds": 12.0,
    "tempo_min": 0.75,                    # usinyooshe zaidi ya hii (inaharibu sauti)
    "tempo_max": 1.35,

    # Mix
    "dialogue_lufs": -16,
    "duck_ratio": 2.5,                    # background ducking wakati wa dialogue
    "sample_rate": 44100,
}

IS_WINDOWS = os.name == "nt"


def find_binary(name: str, windows_fallback: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if IS_WINDOWS and Path(windows_fallback).exists():
        return windows_fallback
    raise FileNotFoundError(
        f"{name} haipatikani. Install ffmpeg au weka path sahihi."
    )


FFMPEG = find_binary("ffmpeg", r"C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe")
FFPROBE = find_binary("ffprobe", r"C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffprobe.exe")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def run_ffmpeg(args: list, desc: str = ""):
    """Run ffmpeg na error reporting halisi (sio silent failure kama v1)."""
    result = subprocess.run([FFMPEG, *args, "-y"], capture_output=True, text=True)
    if result.returncode != 0:
        tail = result.stderr[-1500:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"ffmpeg failed [{desc}]:\n{tail}")
    return result


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def atempo_chain(factor: float) -> str:
    """ffmpeg atempo inakubali 0.5–2.0 per filter; chain kama iko nje ya range."""
    factor = max(0.25, min(4.0, factor))
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.5f}")
    return ",".join(parts)


# ═══════════════════════════════════════════════════════════════
class DubbingPipeline:
    def __init__(self, input_movie: str, output_dir: str):
        self.input_movie = Path(input_movie)
        if not self.input_movie.exists():
            raise FileNotFoundError(f"Movie haipo: {input_movie}")

        self.movie_name = self.input_movie.stem
        self.work = Path(output_dir) / self.movie_name
        self.audio_dir = self.work / "audio"
        self.refs_dir = self.work / "refs"
        self.tts_dir = self.work / "tts"
        self.final_dir = self.work / "final"
        self.ckpt = self.work / "checkpoints"
        for d in (self.audio_dir, self.refs_dir, self.tts_dir,
                  self.final_dir, self.ckpt):
            d.mkdir(parents=True, exist_ok=True)

        self.vocals = self.audio_dir / "vocals.wav"
        self.background = self.audio_dir / "background.wav"

    # ── checkpoint helpers (Kaggle resume) ──────────────────────
    def _save(self, name: str, data):
        with open(self.ckpt / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self, name: str):
        p = self.ckpt / f"{name}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _banner(self, msg: str):
        print(f"\n{'═' * 62}\n {msg}\n{'═' * 62}")

    # ════════════════════════════════════════════════════════════
    # STEP 1 — Extract audio
    # ════════════════════════════════════════════════════════════
    def step1_extract(self) -> Path:
        out = self.audio_dir / "original.wav"
        if out.exists():
            print("STEP 1: original.wav ipo — skipping (resume).")
            return out
        self._banner("STEP 1: Extracting audio")
        run_ffmpeg(["-i", str(self.input_movie), "-vn",
                    "-ar", str(CONFIG["sample_rate"]), "-ac", "2", str(out)],
                   "extract audio")
        print(f"✓ {out}")
        return out

    # ════════════════════════════════════════════════════════════
    # STEP 2 — Demucs separation (vocals vs background)
    # ════════════════════════════════════════════════════════════
    def step2_separate(self, original: Path):
        if self.vocals.exists() and self.background.exists():
            print("STEP 2: separation ipo — skipping (resume).")
            return
        self._banner(f"STEP 2: Demucs ({CONFIG['demucs_model']})")
        sep_dir = self.work / "separated"
        import demucs.separate
        demucs.separate.main([
            "--two-stems", "vocals",
            "-n", CONFIG["demucs_model"],
            "--out", str(sep_dir),
            str(original),
        ])
        src = sep_dir / CONFIG["demucs_model"] / original.stem
        shutil.copy(src / "vocals.wav", self.vocals)
        shutil.copy(src / "no_vocals.wav", self.background)
        print(f"✓ vocals → {self.vocals}\n✓ background → {self.background}")

    # ════════════════════════════════════════════════════════════
    # STEP 3 — Diarization (on CLEAN vocals)
    # ════════════════════════════════════════════════════════════
    def step3_diarize(self) -> dict:
        cached = self._load("speakers")
        if cached:
            print("STEP 3: diarization checkpoint ipo — skipping.")
            return cached
        self._banner("STEP 3: Speaker diarization (pyannote 3.1)")

        import torch
        import soundfile as sf
        from pyannote.audio import Pipeline as Pyannote

        pipe = Pyannote.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN)
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))

        data, sr = sf.read(str(self.vocals))
        wav = data.reshape(1, -1) if data.ndim == 1 else data.T
        diar = pipe({"waveform": torch.tensor(wav, dtype=torch.float32),
                     "sample_rate": sr})

        speakers: dict = {}
        for segment, _, spk in diar.itertracks(yield_label=True):
            speakers.setdefault(spk, []).append(
                {"start": float(segment.start), "end": float(segment.end)})

        if not speakers:
            speakers["SPEAKER_00"] = [{"start": 0.0, "end": probe_duration(self.vocals)}]

        print(f"✓ Speakers: {len(speakers)} → {list(speakers)}")
        self._save("speakers", speakers)
        return speakers

    # ════════════════════════════════════════════════════════════
    # STEP 4 — Transcription: faster-whisper large-v3 + VAD
    #          + hallucination filtering  (FIX #3, #4)
    # ════════════════════════════════════════════════════════════
    def step4_transcribe(self, speakers: dict) -> list:
        cached = self._load("segments")
        if cached:
            print("STEP 4: transcript checkpoint ipo — skipping.")
            return cached
        self._banner(f"STEP 4: faster-whisper {CONFIG['whisper_model']} + VAD")

        import torch
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = CONFIG["whisper_compute"] if device == "cuda" else "int8"
        model = WhisperModel(CONFIG["whisper_model"], device=device,
                             compute_type=compute)

        raw_segments, info = model.transcribe(
            str(self.vocals),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms":
                            CONFIG["vad_min_silence_ms"]},
            condition_on_previous_text=False,   # inazuia hallucination loops
            word_timestamps=True,
        )
        print(f"  Detected language: {info.language} "
              f"(prob {info.language_probability:.2f})")

        def best_speaker(start: float, end: float) -> str:
            best, overlap = None, 0.0
            for spk, segs in speakers.items():
                for s in segs:
                    ov = max(0.0, min(end, s["end"]) - max(start, s["start"]))
                    if ov > overlap:
                        overlap, best = ov, spk
            return best or next(iter(speakers))

        segments, dropped, prev_text = [], 0, ""
        for seg in raw_segments:
            text = seg.text.strip()
            # ── hallucination filters ──
            if not text:
                continue
            if seg.no_speech_prob > CONFIG["max_no_speech_prob"]:
                dropped += 1
                continue
            if seg.avg_logprob < CONFIG["min_avg_logprob"]:
                dropped += 1
                continue
            if seg.compression_ratio > CONFIG["max_compression_ratio"]:
                dropped += 1
                continue
            if text == prev_text and len(text) > 12:   # loop repetition
                dropped += 1
                continue
            prev_text = text

            segments.append({
                "id": len(segments),
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "duration": round(float(seg.end - seg.start), 3),
                "text": text,
                "speaker": best_speaker(seg.start, seg.end),
            })

        print(f"✓ {len(segments)} segments | {dropped} hallucinations dropped")
        self._save("segments", segments)
        return segments

    # ════════════════════════════════════════════════════════════
    # STEP 5 — Translation: batched, scene-context, JSON-strict
    # ════════════════════════════════════════════════════════════
    def step5_translate(self, segments: list) -> list:
        cached = self._load("translated")
        if cached:
            print("STEP 5: translation checkpoint ipo — skipping.")
            return cached
        self._banner("STEP 5: Groq translation (scene-context batches)")

        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY haijawekwa kwenye environment.")
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        SYSTEM = (
            "Wewe ni mtafsiri mkuu wa filamu (dubbing). Tafsiri dialogue kuwa "
            "Kiswahili cha mazungumzo ya kawaida ya Tanzania (mtaani lakini safi). "
            "KANUNI:\n"
            "1. Hisia za mhusika zibaki kamili: hasira, vicheko, hofu, mapenzi.\n"
            "2. Idadi ya silabi za tafsiri iwe KARIBU sawa na asili (lip-sync). "
            "Kama lazima, fupisha bila kupoteza maana.\n"
            "3. Majina ya watu na mahali yabaki kama yalivyo.\n"
            "4. Rudisha JSON array TUPU TU: "
            '[{"id": <int>, "sw": "<tafsiri>"}] — bila maelezo, bila markdown.'
        )

        def translate_batch(batch: list, attempt: int = 0) -> dict:
            lines = "\n".join(
                f'{{"id": {s["id"]}, "speaker": "{s["speaker"]}", '
                f'"text": {json.dumps(s["text"])}}}' for s in batch)
            try:
                resp = client.chat.completions.create(
                    model=CONFIG["groq_model"],
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": lines}],
                    max_tokens=4000, temperature=0.3,
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                return {item["id"]: item["sw"] for item in parsed}
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    return translate_batch(batch, attempt + 1)
                print(f"  ! Batch failed ({e}) — fallback per-line")
                out = {}
                for s in batch:
                    try:
                        r = client.chat.completions.create(
                            model=CONFIG["groq_model"],
                            messages=[{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content":
                                       f'[{{"id": {s["id"]}, "text": '
                                       f'{json.dumps(s["text"])}}}]'}],
                            max_tokens=300, temperature=0.3)
                        raw = (r.choices[0].message.content
                               .replace("```json", "").replace("```", "").strip())
                        out[s["id"]] = json.loads(raw)[0]["sw"]
                    except Exception:
                        out[s["id"]] = s["text"]   # last resort: keep original
                    time.sleep(0.5)
                return out

        translated, bs = [], CONFIG["translate_batch_size"]
        for i in range(0, len(segments), bs):
            batch = segments[i:i + bs]
            mapping = translate_batch(batch)
            for s in batch:
                translated.append({**s, "swahili": mapping.get(s["id"], s["text"])})
            done = min(i + bs, len(segments))
            print(f"  [{done}/{len(segments)}] "
                  f"{translated[-1]['swahili'][:55]}")
            time.sleep(0.5)   # free-tier rate limit cushion

        self._save("translated", translated)
        print(f"✓ Translated {len(translated)} segments")
        return translated

    # ════════════════════════════════════════════════════════════
    # STEP 6 — TTS: PER-SEGMENT prosody cloning  (FIX #1, #2, #5)
    # ════════════════════════════════════════════════════════════
    def step6_tts(self, translated: list, speakers: dict) -> list:
        cached = self._load("tts_done")
        if cached:
            print("STEP 6: TTS checkpoint ipo — skipping.")
            return cached
        self._banner("STEP 6: XTTS-v2 — per-segment emotion cloning")

        import torch
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if torch.cuda.is_available():
            tts.to("cuda")

        vocals_dur = probe_duration(self.vocals)

        # fallback ref kwa kila speaker: segment ndefu zaidi, CLEAN vocals
        speaker_fallback: dict = {}
        for spk, segs in speakers.items():
            best = max(segs, key=lambda s: s["end"] - s["start"])
            start = max(0.0, best["start"])
            dur = min(best["end"] - start, CONFIG["max_ref_seconds"])
            ref = self.refs_dir / f"fallback_{spk}.wav"
            if not ref.exists():
                run_ffmpeg(["-i", str(self.vocals), "-ss", f"{start:.3f}",
                            "-t", f"{dur:.3f}", "-ar", "22050", "-ac", "1",
                            str(ref)], f"fallback ref {spk}")
            speaker_fallback[spk] = ref

        def segment_ref(seg: dict, idx: int) -> Path:
            """Kata reference kutoka SEGMENT YENYEWE (clean vocals) —
            hii ndiyo inayohamisha EMOTION ya line husika kwenye TTS."""
            start, end = seg["start"], seg["end"]
            dur = end - start
            if dur < CONFIG["min_ref_seconds"]:
                pad = (CONFIG["min_ref_seconds"] - dur) / 2
                start = max(0.0, start - pad)
                end = min(vocals_dur, start + CONFIG["min_ref_seconds"])
            dur = min(end - start, CONFIG["max_ref_seconds"])
            if dur < CONFIG["min_ref_seconds"] - 0.1:
                return speaker_fallback[seg["speaker"]]
            ref = self.refs_dir / f"ref_{idx:04d}.wav"
            run_ffmpeg(["-i", str(self.vocals), "-ss", f"{start:.3f}",
                        "-t", f"{dur:.3f}", "-ar", "22050", "-ac", "1",
                        str(ref)], f"ref {idx}")
            return ref

        total = len(translated)
        for i, seg in enumerate(translated):
            out = self.tts_dir / f"seg_{i:04d}.wav"
            fitted = self.tts_dir / f"fit_{i:04d}.wav"
            if fitted.exists():
                seg["tts_path"] = str(fitted)
                continue
            try:
                ref = segment_ref(seg, i)
                tts.tts_to_file(
                    text=seg["swahili"],
                    speaker_wav=str(ref),
                    language=CONFIG["xtts_lang"],   # "es" — vokali safi za KiSwahili
                    file_path=str(out),
                )
                # ── FIX #4: duration fit (atempo) ──
                tts_dur = probe_duration(out)
                target = max(seg["duration"], 0.3)
                factor = tts_dur / target
                factor = max(CONFIG["tempo_min"], min(CONFIG["tempo_max"], factor))
                run_ffmpeg(["-i", str(out),
                            "-filter:a", atempo_chain(factor),
                            "-ar", str(CONFIG["sample_rate"]), str(fitted)],
                           f"tempo fit {i}")
                seg["tts_path"] = str(fitted)
                print(f"  [{i + 1}/{total}] ✓ {seg['speaker']} "
                      f"(x{factor:.2f}) {seg['swahili'][:42]}")
            except Exception as e:
                print(f"  [{i + 1}/{total}] ✗ {e}")

        self._save("tts_done", translated)
        return translated

    # ════════════════════════════════════════════════════════════
    # STEP 7 — Mix: loudnorm dialogue + ducked background  (FIX #6)
    # ════════════════════════════════════════════════════════════
    def step7_mix(self, translated: list) -> Path:
        self._banner("STEP 7: Final mix + mux")
        movie_dur = probe_duration(self.input_movie)
        valid = [s for s in translated if "tts_path" in s]
        if not valid:
            raise RuntimeError("Hakuna TTS segments — step 6 ilifeli yote.")

        # 7a. Dialogue bus: place kila segment kwa adelay, kisha loudnorm
        dialogue = self.audio_dir / "dialogue_sw.wav"
        CHUNK = 100   # epuka ffmpeg command-line limits kwa movie ndefu
        chunk_files = []
        for c in range(0, len(valid), CHUNK):
            chunk = valid[c:c + CHUNK]
            inputs, filters = [], []
            for j, seg in enumerate(chunk):
                inputs += ["-i", seg["tts_path"]]
                delay = int(seg["start"] * 1000)
                filters.append(f"[{j}]adelay={delay}|{delay}[s{j}]")
            mix_in = "".join(f"[s{j}]" for j in range(len(chunk)))
            filters.append(
                f"{mix_in}amix=inputs={len(chunk)}:normalize=0:"
                f"duration=longest[out]")
            cf = self.audio_dir / f"dchunk_{c:05d}.wav"
            run_ffmpeg([*inputs, "-filter_complex", ";".join(filters),
                        "-map", "[out]", "-t", str(movie_dur), str(cf)],
                       f"dialogue chunk {c}")
            chunk_files.append(cf)

        if len(chunk_files) == 1:
            raw_dialogue = chunk_files[0]
        else:
            inputs, n = [], len(chunk_files)
            for cf in chunk_files:
                inputs += ["-i", str(cf)]
            raw_dialogue = self.audio_dir / "dialogue_raw.wav"
            run_ffmpeg([*inputs, "-filter_complex",
                        "".join(f"[{k}]" for k in range(n)) +
                        f"amix=inputs={n}:normalize=0[out]",
                        "-map", "[out]", str(raw_dialogue)], "merge chunks")

        run_ffmpeg(["-i", str(raw_dialogue), "-af",
                    f"loudnorm=I={CONFIG['dialogue_lufs']}:TP=-1.5:LRA=11",
                    "-ar", str(CONFIG["sample_rate"]), str(dialogue)],
                   "dialogue loudnorm")

        # 7b. Background full + sidechain ducking under dialogue
        final_audio = self.audio_dir / "final_audio.wav"
        run_ffmpeg([
            "-i", str(self.background), "-i", str(dialogue),
            "-filter_complex",
            ("[1:a]asplit=2[dkey][dmix];"
             f"[0:a][dkey]sidechaincompress=threshold=0.06:"
             f"ratio={CONFIG['duck_ratio']}:attack=15:release=350[bg];"
             "[bg][dmix]amix=inputs=2:normalize=0:duration=first[out]"),
            "-map", "[out]", "-ar", str(CONFIG["sample_rate"]),
            str(final_audio)], "final mix")

        # 7c. Mux na video (video stream copy — hakuna re-encode)
        final = self.final_dir / f"{self.movie_name}_KISWAHILI.mp4"
        run_ffmpeg(["-i", str(self.input_movie), "-i", str(final_audio),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(final)], "mux")
        print(f"\n✅ FINAL MOVIE: {final}")
        return final

    # ════════════════════════════════════════════════════════════
    def run(self) -> Path:
        t0 = time.time()
        self._banner(f"SWAHILI VOICE MASTER v2.0 — {self.movie_name}")
        original = self.step1_extract()
        self.step2_separate(original)
        speakers = self.step3_diarize()
        segments = self.step4_transcribe(speakers)
        translated = self.step5_translate(segments)
        translated = self.step6_tts(translated, speakers)
        final = self.step7_mix(translated)
        mins = (time.time() - t0) / 60
        self._banner(f"✅ DONE in {mins:.1f} min → {final}")
        return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Swahili Voice Master v2.0")
    ap.add_argument("--input", required=True, help="Path ya movie (mp4/mkv)")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()
    DubbingPipeline(args.input, args.output).run()
