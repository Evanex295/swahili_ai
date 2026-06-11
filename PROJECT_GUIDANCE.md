# SWAHILI VOICE MASTER — PROJECT GUIDANCE v2.0

Mwongozo kamili: setup ya VS Code, Claude Code, GitHub push, Kaggle workflow,
na roadmap ya kufikia platform ya biashara.

---

## 1. MUUNDO WA PROJECT (weka hivi kwenye VS Code)

```
swahili-voice-master/
├── pipeline_v2.py          # Pipeline kuu (mpya — tumia hii)
├── requirements.txt
├── PROJECT_GUIDANCE.md     # Hii file
├── .env                    # API keys (USIIPUSH GitHub!)
├── .gitignore
├── movies_in/              # Movies za kutafsiri (local testing)
├── output/                 # Matokeo
└── legacy/                 # Code ya zamani (pipeline.py, step1-4)
```

**Hatua za ku-setup VS Code:**

1. Fungua VS Code → `File → Open Folder` → tengeneza folder `swahili-voice-master`
2. Weka `pipeline_v2.py` na `requirements.txt` ndani
3. Hamisha files za zamani (`pipeline.py`, `step1_whisper.py`, n.k.) kwenye
   folder `legacy/` — usizifute, ni reference
4. Tengeneza `.env`:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxx
   HF_TOKEN=hf_xxxxxxxxxxxx
   ```
5. Tengeneza `.gitignore`:
   ```
   .env
   movies_in/
   output/
   __pycache__/
   *.wav
   *.mp4
   ```
6. Terminal ndani ya VS Code (`Ctrl+` `):
   ```
   python -m venv venv
   venv\Scripts\activate        (Windows)
   pip install -r requirements.txt
   ```

**HF_TOKEN:** nenda huggingface.co → Settings → Access Tokens → New Token,
kisha **accept terms** za models hizi mbili (lazima, vinginevyo diarization
itafeli): `pyannote/speaker-diarization-3.1` na `pyannote/segmentation-3.0`.

**GROQ_API_KEY:** console.groq.com → API Keys (free tier inatosha kuanza).

---

## 2. KUTUMIA CLAUDE CODE KWENYE PROJECT HII

Claude Code ni terminal agent — inafanya kazi DIRECTLY kwenye files zako.

**Install (Windows — inahitaji Node.js 18+):**
```
npm install -g @anthropic-ai/claude-code
```

**Workflow:**
```
cd swahili-voice-master
claude
```

**Prompts za mfano zinazofaa project hii:**

| Lengo | Prompt |
|---|---|
| Kuelewa code | `Explain how step6_tts achieves per-segment emotion cloning` |
| Ku-debug | `Pipeline failed at step 7 with this error: <paste>. Fix it` |
| Feature mpya | `Add a --resume-from flag that lets me restart from any step` |
| Subtitle export | `Add SRT subtitle export from the translated checkpoint` |
| Web phase | `Scaffold a FastAPI backend with an /upload endpoint that queues movies for the pipeline` |

**Kanuni:** Claude Code inaona files zote kwenye folder — kwa hiyo `.env`
iwe kwenye `.gitignore` na uwe makini na secrets. Tumia `/init` mara ya
kwanza ili itengeneze `CLAUDE.md` ya project context.

---

## 3. KU-PUSH GITHUB (na kuilinda code yako)

```
git init
git add .
git commit -m "Swahili Voice Master v2.0 - per-segment emotion cloning pipeline"
```

GitHub → New Repository → **PRIVATE** (hii ni biashara yako — usiweke public).

```
git remote add origin https://github.com/USERNAME/swahili-voice-master.git
git branch -M main
git push -u origin main
```

**Updates za kawaida:**
```
git add -A
git commit -m "fix: tempo clamp for short segments"
git push
```

**Verify kabla ya kila push:** `.env` HAIPO kwenye `git status`. Ikionekana:
```
git rm --cached .env
```

---

## 4. KAGGLE WORKFLOW (GPU ya bure — speed ya dubbing)

Kaggle inakupa **T4 x2 / P100 GPU, masaa ~30 kwa wiki, bure.** Hii ndiyo
engine yako ya production mpaka utakapopata mapato.

### Setup ya Notebook

1. kaggle.com → Create → New Notebook
2. Settings (kulia): **Accelerator → GPU T4 x2**, Internet → **ON**
3. Add-ons → Secrets → ongeza `GROQ_API_KEY` na `HF_TOKEN`
4. Upload movie: Datasets → New Dataset → upload mp4 → attach kwenye notebook

### Notebook Cells

**Cell 1 — Install (dakika ~5):**
```python
!pip install -q faster-whisper demucs pyannote.audio TTS groq
!apt-get -qq install -y ffmpeg
```

**Cell 2 — Secrets:**
```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["GROQ_API_KEY"] = s.get_secret("GROQ_API_KEY")
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")
os.environ["COQUI_TOS_AGREED"] = "1"   # XTTS license auto-accept
```

**Cell 3 — Pakia pipeline (kutoka GitHub yako private, tumia token):**
```python
!git clone https://USERNAME:ghp_TOKEN@github.com/USERNAME/swahili-voice-master.git
%cd swahili-voice-master
```

**Cell 4 — Run:**
```python
!python pipeline_v2.py \
    --input /kaggle/input/YOUR-DATASET/movie.mp4 \
    --output /kaggle/working/out
```

**Cell 5 — Download result:**
```python
from IPython.display import FileLink
import glob
final = glob.glob("/kaggle/working/out/**/final/*.mp4", recursive=True)[0]
FileLink(final)
```

### Kaggle Tips Muhimu

- **Session limit ni masaa 12** — pipeline v2 ina checkpoints kila step.
  Session ikikatika, run tena cell ile ile: itaskip steps zilizokamilika.
- Movie ya saa 2: tegemea **masaa 3–6** kwenye T4 (Demucs + TTS ndizo nzito).
- Series: process episode moja kwa session moja, panga ratiba ya wiki.
- `htdemucs_ft` ni bora lakini polepole; ukitaka speed badilisha CONFIG
  kuwa `"htdemucs"` (4x faster, quality bado nzuri).

---

## 5. FIXES ZA v2.0 — TECHNICAL SUMMARY

| # | Tatizo la v1 | Fix ya v2 |
|---|---|---|
| 1 | Hakuna emotion — sample 1 ya 20s per speaker | **Per-segment prosody cloning**: kila line ina-clone kutoka audio ya segment yenyewe. Hasira → hasira, kicheko → kicheko |
| 2 | `language="en"` kwa Kiswahili — matamshi mabovu | `language="es"` — Spanish ina vokali a-e-i-o-u safi kama Kiswahili + rolled R |
| 3 | Maneno ya uongo (hallucinations) | faster-whisper **large-v3** + Silero VAD + filters tatu (no_speech_prob, avg_logprob, compression_ratio) |
| 4 | Segments zina-overlap, timing broken | atempo duration-fitting kila segment (clamp 0.75–1.35) |
| 5 | Voice refs zilikatwa kwenye original (music ndani) | Refs zote kutoka **separated vocals** — clean |
| 6 | Background haisikiki | Dialogue loudnorm -16 LUFS + background full + sidechain ducking |
| 7 | Kaggle session ikikatika unaanza upya | Checkpoint/resume kila step |
| 8 | ffmpeg errors zilikuwa silent | Error reporting kamili kila call |

### Limitation moja ya kweli (sema ukweli kwa wateja wako)

XTTS-v2 haina native Swahili. Trick ya `"es"` + per-segment cloning
inaboresha SANA, lakini accent itabaki kidogo. **Fix ya kudumu = fine-tune
XTTS-v2 kwa Swahili data** (Mozilla Common Voice Swahili ina masaa mengi ya
bure). Hii inawezekana kwenye Kaggle GPU — ni Phase 2 project ya wiki 2–4,
na ikifanikiwa inakuwa **competitive moat yako halisi**: hakuna mtu mwingine
East Africa mwenye Swahili voice-cloning TTS model. Hilo ndilo product
ambalo Azam wanaweza kulipia, sio pipeline ya open-source tools peke yake.

---

## 6. ROADMAP YA BIASHARA (Phases)

**Phase 1 — SASA (wiki 1–2): Prove quality**
- Run v2.0 kwenye movie 2–3 (action + drama + Korean)
- Linganisha na output ya zamani — demo reel ya before/after
- Hii ndiyo jibu lako kwa wale waliokuambia ni "nonsense": waonyeshe output

**Phase 2 (mwezi 1–2): Swahili XTTS fine-tune**
- Download Common Voice Swahili dataset
- Fine-tune XTTS-v2 kwenye Kaggle (kuna recipes za wazi za Coqui)
- Hii inaondoa accent issue kabisa = broadcast quality

**Phase 3 (mwezi 2–3): Website + delivery platform**
- FastAPI backend + queue (movies zina-process Kaggle/server)
- Frontend: upload → status → download (unaweza kutumia stack yako ya
  rochveld.com — PHP/React unazijua tayari)
- Payment: M-Pesa/Tigo Pesa integration (Flutterwave/Selcom APIs)
- Catalog page ya "DONE MOVIES" — preview clips, sio full movies

**Phase 4: Partnerships (Azam TV, media houses)**
- Nenda na DEMO, sio pitch deck. Episode moja kamili ya Korean drama
  kwa Kiswahili, quality ya broadcast
- Model ya bei: per-episode fee au licensing ya teknolojia
- **MUHIMU KISHERIA:** kabla ya ku-pitch Azam au kuuza movies kwenye
  website, hakikisha movies unazotafsiri una HAKI ya kuzitafsiri.
  Dubbing ya movie ya mtu bila license ni copyright infringement —
  Azam wenyewe ni rights holders, watakuuliza hili siku ya kwanza.
  Model salama: uza TEKNOLOJIA/HUDUMA (wao walete content yao,
  wewe u-dub) badala ya kuuza movies za watu. Hii pia ni model bora
  ya biashara — recurring revenue badala ya one-off sales.

---

## 7. QUICK COMMANDS REFERENCE

```
# Local run (Windows)
python pipeline_v2.py --input "movies_in\film.mp4" --output "output"

# Kaggle run
!python pipeline_v2.py --input /kaggle/input/ds/film.mp4 --output /kaggle/working/out

# Resume baada ya crash — run command ile ile, checkpoints zinashika

# Badilisha quality/speed: edit CONFIG juu ya pipeline_v2.py
#   demucs_model: "htdemucs" = speed, "htdemucs_ft" = quality
#   whisper_compute: "int8" kama GPU memory inaisha
```
