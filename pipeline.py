import os
import json
import subprocess
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
FFMPEG = r'C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe'
FFPROBE = r'C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffprobe.exe'
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

class MovieTranslationPipeline:
    def __init__(self, input_movie: str, output_dir: str, target_language: str = "sw"):
        self.input_movie = Path(input_movie)
        self.output_dir = Path(output_dir)
        self.target_language = target_language
        self.movie_name = self.input_movie.stem
        
        # Directories
        self.work_dir = self.output_dir / self.movie_name
        self.audio_dir = self.work_dir / "audio"
        self.separated_dir = self.work_dir / "separated"
        self.segments_dir = self.work_dir / "segments"
        self.tts_dir = self.work_dir / "tts"
        self.final_dir = self.work_dir / "final"
        
        for d in [self.audio_dir, self.separated_dir, self.segments_dir, 
                  self.tts_dir, self.final_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"Pipeline initialized for: {self.movie_name}")
        print(f"Working directory: {self.work_dir}")

    # ═══════════════════════════════════════════════════════
    # STEP 1: EXTRACT AUDIO
    # ═══════════════════════════════════════════════════════
    def step1_extract_audio(self):
        print("\n" + "="*60)
        print("STEP 1: Extracting audio from movie...")
        print("="*60)
        
        audio_path = self.audio_dir / "original.wav"
        subprocess.run([
            FFMPEG, "-i", str(self.input_movie),
            "-vn", "-ar", "44100", "-ac", "2",
            "-y", str(audio_path)
        ], check=True, capture_output=True)
        
        print(f"✓ Audio extracted: {audio_path}")
        return audio_path

    # ═══════════════════════════════════════════════════════
    # STEP 2: SEPARATE DIALOGUE FROM BACKGROUND (Demucs)
    # ═══════════════════════════════════════════════════════
    def step2_separate_audio(self, audio_path: Path):
        print("\n" + "="*60)
        print("STEP 2: Separating dialogue from background music...")
        print("Using Demucs htdemucs_ft model (highest quality)...")
        print("="*60)
        
        import demucs.separate
        
        # Use htdemucs_ft — best model for voice/music separation
        demucs.separate.main([
            "--two-stems", "vocals",  # Separate vocals from everything else
            "-n", "htdemucs_ft",      # Best quality model
            "--out", str(self.separated_dir),
            str(audio_path)
        ])
        
        # Find output files
        separated_folder = self.separated_dir / "htdemucs_ft" / "original"
        vocals_path = separated_folder / "vocals.wav"
        background_path = separated_folder / "no_vocals.wav"
        
        print(f"✓ Vocals (dialogue): {vocals_path}")
        print(f"✓ Background (music+effects): {background_path}")
        
        return vocals_path, background_path

    # ═══════════════════════════════════════════════════════
    # STEP 3: SPEAKER DIARIZATION
    # ═══════════════════════════════════════════════════════
    def step3_diarization(self, vocals_path: Path):
        print("\n" + "="*60)
        print("STEP 3: Speaker diarization...")
        print("="*60)
        
        from pyannote.audio import Pipeline as PyannotePipeline
        
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        
        # Load audio properly
        data, sample_rate = sf.read(str(vocals_path))
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T
        waveform = torch.tensor(data, dtype=torch.float32)
        audio = {"waveform": waveform, "sample_rate": sample_rate}
        
        diarization = pipeline(audio)
        
        # Parse results
        speakers = {}
        try:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append({
                    "start": segment.start,
                    "end": segment.end
                })
        except:
            # Fallback for new pyannote API
            for item in diarization:
                speaker = str(item[2]) if len(item) > 2 else "SPEAKER_00"
                seg = item[0]
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append({
                    "start": float(seg.start),
                    "end": float(seg.end)
                })
        
        print(f"✓ Found {len(speakers)} speakers: {list(speakers.keys())}")
        
        # Extract voice sample for each speaker
        speaker_samples = {}
        for speaker, segments in speakers.items():
            # Get longest segment for best voice sample
            best = max(segments, key=lambda x: x["end"] - x["start"])
            duration = min(best["end"] - best["start"], 20.0)
            
            sample_path = self.audio_dir / f"sample_{speaker}.wav"
            subprocess.run([
                FFMPEG, "-i", str(self.input_movie),
                "-ss", str(best["start"]),
                "-t", str(duration),
                "-vn", "-ar", "22050", "-ac", "1",
                "-y", str(sample_path)
            ], capture_output=True)
            
            speaker_samples[speaker] = str(sample_path)
            print(f"  ✓ Voice sample extracted for {speaker}: {duration:.1f}s")
        
        return speakers, speaker_samples

    # ═══════════════════════════════════════════════════════
    # STEP 4: TRANSCRIPTION (Whisper)
    # ═══════════════════════════════════════════════════════
    def step4_transcribe(self, vocals_path: Path, speakers: dict):
        print("\n" + "="*60)
        print("STEP 4: Transcribing with Whisper...")
        print("="*60)
        
        import whisper
        import os
        os.environ["PATH"] = r"C:\ffmpeg\ffmpeg-8.1-essentials_build\bin" + os.pathsep + os.environ["PATH"]
        
        model = whisper.load_model("medium")
        result = model.transcribe(
            str(vocals_path),
            word_timestamps=True,
            verbose=False
        )
        
        def get_speaker(start, end):
            best_speaker = None
            best_overlap = 0
            for speaker, segs in speakers.items():
                for seg in segs:
                    overlap = max(0, min(end, seg["end"]) - max(start, seg["start"]))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker
            return best_speaker or list(speakers.keys())[0]
        
        segments = []
        for seg in result["segments"]:
            speaker = get_speaker(seg["start"], seg["end"])
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "speaker": speaker
            })
        
        transcript_path = self.work_dir / "transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Transcribed {len(segments)} segments")
        return segments

    # ═══════════════════════════════════════════════════════
    # STEP 5: TRANSLATION (Groq - Cinematic Context)
    # ═══════════════════════════════════════════════════════
    def step5_translate(self, segments: list):
        print("\n" + "="*60)
        print("STEP 5: Translating with cinematic context...")
        print("="*60)
        
        from groq import Groq
        import time
        
        client = Groq(api_key=GROQ_API_KEY)
        
        # Build context from surrounding segments
        def translate_with_context(segment, all_segments, idx):
            # Get previous and next segments for context
            prev = all_segments[max(0, idx-2):idx]
            next_segs = all_segments[idx+1:min(len(all_segments), idx+3)]
            
            context_prev = "\n".join([f"[{s['speaker']}]: {s['text']}" for s in prev])
            context_next = "\n".join([f"[{s['speaker']}]: {s['text']}" for s in next_segs])
            
            prompt = f"""Wewe ni mtafsiri wa filamu wa professional.
Tafsiri dialogue hii kuwa Kiswahili cha mazungumzo ya kawaida ya Tanzania.

MUKTADHA WA AWALI:
{context_prev}

TAFSIRI HII (Mhusika: {segment['speaker']}):
"{segment['text']}"

MUKTADHA UNAOFUATA:
{context_next}

KANUNI ZA PROFESSIONAL:
1. Kiswahili cha mazungumzo ya kawaida — sio rasmi
2. Hisia zibaki: hasira, furaha, hofu, upole
3. Urefu wa sentensi uwe KARIBU SAWA na asili (muhimu kwa lip sync)
4. Jibu na TAFSIRI TU — bila maelezo"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        
        translated = []
        total = len(segments)
        
        for i, seg in enumerate(segments):
            sw_text = translate_with_context(seg, segments, i)
            translated.append({
                **seg,
                "swahili": sw_text
            })
            print(f"[{i+1}/{total}] [{seg['speaker']}] {seg['text'][:35]}...")
            print(f"          → {sw_text[:50]}")
            time.sleep(0.2)
        
        translation_path = self.work_dir / "translation.json"
        with open(translation_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Translated {total} segments")
        return translated

    # ═══════════════════════════════════════════════════════
    # STEP 6: VOICE CLONING (XTTS per speaker)
    # ═══════════════════════════════════════════════════════
    def step6_tts(self, translated: list, speaker_samples: dict):
        print("\n" + "="*60)
        print("STEP 6: Generating cloned voices per speaker...")
        print("="*60)
        
        from TTS.api import TTS
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        total = len(translated)
        
        for i, seg in enumerate(translated):
            speaker = seg["speaker"]
            sample = speaker_samples.get(speaker, list(speaker_samples.values())[0])
            output_path = self.tts_dir / f"seg_{i:04d}.wav"
            
            try:
                tts.tts_to_file(
                    text=seg["swahili"],
                    speaker_wav=sample,
                    language="en",
                    file_path=str(output_path)
                )
                # Store actual duration of generated audio
                data, sr = sf.read(str(output_path))
                translated[i]["tts_duration"] = len(data) / sr
                translated[i]["tts_path"] = str(output_path)
                print(f"[{i+1}/{total}] ✓ {speaker}: {seg['swahili'][:45]}")
            except Exception as e:
                print(f"[{i+1}/{total}] ✗ Error: {e}")
        
        return translated

    # ═══════════════════════════════════════════════════════
    # STEP 7: FINAL MIX — Precise timing + Background preservation
    # ═══════════════════════════════════════════════════════
    def step7_final_mix(self, translated: list, background_path: Path):
        print("\n" + "="*60)
        print("STEP 7: Final mix — background + translated dialogue...")
        print("="*60)
        
        # Get movie duration
        result = subprocess.run([
            FFPROBE, "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.input_movie)
        ], capture_output=True, text=True)
        movie_duration = float(result.stdout.strip())
        
        # Build silence base track
        silence_path = self.audio_dir / "silence.wav"
        subprocess.run([
            FFMPEG, "-f", "lavfi", "-i",
            f"anullsrc=r=24000:cl=mono",
            "-t", str(movie_duration),
            "-y", str(silence_path)
        ], capture_output=True, check=True)
        
        # Build FFmpeg filter for precise placement of each TTS segment
        inputs = ["-i", str(silence_path)]
        filter_parts = []
        
        valid_segments = [s for s in translated if "tts_path" in s]
        
        for i, seg in enumerate(valid_segments):
            inputs += ["-i", seg["tts_path"]]
            delay_ms = int(seg["start"] * 1000)
            filter_parts.append(
                f"[{i+1}]adelay={delay_ms}|{delay_ms}[s{i}]"
            )
        
        mix_inputs = "".join(f"[s{i}]" for i in range(len(valid_segments)))
        filter_parts.append(
            f"[0]{mix_inputs}amix=inputs={len(valid_segments)+1}:normalize=0[dialogue_sw]"
        )
        
        dialogue_sw_path = self.audio_dir / "dialogue_sw.wav"
        
        cmd = [FFMPEG] + inputs + [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[dialogue_sw]",
            "-y", str(dialogue_sw_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Final mix: background (full volume) + SW dialogue
        final_audio = self.audio_dir / "final_audio.wav"
        subprocess.run([
            FFMPEG,
            "-i", str(background_path),
            "-i", str(dialogue_sw_path),
            "-filter_complex", "[0][1]amix=inputs=2:normalize=0[out]",
            "-map", "[out]",
            "-y", str(final_audio)
        ], check=True, capture_output=True)
        
        # Combine with video
        final_movie = self.final_dir / f"{self.movie_name}_kiswahili.mp4"
        subprocess.run([
            FFMPEG,
            "-i", str(self.input_movie),
            "-i", str(final_audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest", "-y",
            str(final_movie)
        ], check=True, capture_output=True)
        
        print(f"✓ Final movie: {final_movie}")
        return final_movie

    # ═══════════════════════════════════════════════════════
    # RUN FULL PIPELINE
    # ═══════════════════════════════════════════════════════
    def run(self):
        print("\n" + "="*60)
        print("SWAHILI VOICE MASTER — PRODUCTION PIPELINE")
        print(f"Movie: {self.movie_name}")
        print("="*60)
        
        audio = self.step1_extract_audio()
        vocals, background = self.step2_separate_audio(audio)
        speakers, speaker_samples = self.step3_diarization(vocals)
        segments = self.step4_transcribe(vocals, speakers)
        translated = self.step5_translate(segments)
        translated = self.step6_tts(translated, speaker_samples)
        final = self.step7_final_mix(translated, background)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print(f"Output: {final}")
        print("="*60)
        return final


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    pipeline = MovieTranslationPipeline(
        input_movie=r"C:\swahili_ai_project\filamu_asili\Dominance.mp4",
        output_dir=r"C:\swahili_ai_project\matokeo",
        target_language="sw"
    )
    pipeline.run()