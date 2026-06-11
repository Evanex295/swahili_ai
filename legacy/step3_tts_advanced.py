import json
import os
import subprocess
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from TTS.api import TTS

TAFSIRI = r'C:\swahili_ai_project\tafsiri\tafsiri_kiswahili.json'
FILAMU = r'C:\swahili_ai_project\filamu_asili\Dominance.mp4'
SAMPULI_JUMLA = r'C:\swahili_ai_project\sampuli_sauti.wav'
FOLDA_SAUTI = r'C:\swahili_ai_project\sauti_vipande'
FOLDA_SAMPULI = r'C:\swahili_ai_project\sampuli_wahusika'
FFMPEG = r'C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe'
HF_TOKEN = os.environ.get("HF_TOKEN")

os.makedirs(FOLDA_SAUTI, exist_ok=True)
os.makedirs(FOLDA_SAMPULI, exist_ok=True)

# ═══════════════════════════════════════════
# HATUA 1: Tambua wahusika
# ═══════════════════════════════════════════
print('='*50)
print('HATUA 1: Inatambua wahusika...')
print('='*50)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

# Soma sampuli kwa soundfile
data_audio, sample_rate = sf.read(SAMPULI_JUMLA)
if len(data_audio.shape) == 1:
    data_audio = data_audio.reshape(1, -1)
else:
    data_audio = data_audio.T
waveform = torch.tensor(data_audio, dtype=torch.float32)
audio = {'waveform': waveform, 'sample_rate': sample_rate}

diarization = pipeline(audio)

# Hifadhi muda wa kila mhusika - toleo jipya la pyannote
wahusika = {}
print('\nInasoma matokeo ya diarization...')

# Jaribu njia zote za kusoma matokeo
try:
    # Njia ya toleo jipya
    for segment, track, speaker in diarization.itertracks(yield_label=True):
        if speaker not in wahusika:
            wahusika[speaker] = []
        wahusika[speaker].append({
            'start': segment.start,
            'end': segment.end
        })
except AttributeError:
    try:
        # Njia mbadala
        for item in diarization:
            segment = item[0]
            speaker = item[2]
            if speaker not in wahusika:
                wahusika[speaker] = []
            wahusika[speaker].append({
                'start': segment.start,
                'end': segment.end
            })
    except Exception:
        # Njia ya mwisho - tumia DataFrame
        df = diarization.rename(columns={0: 'start', 1: 'end', 2: 'speaker'}) if hasattr(diarization, 'rename') else None
        if df is not None:
            for _, row in df.iterrows():
                speaker = str(row['speaker'])
                if speaker not in wahusika:
                    wahusika[speaker] = []
                wahusika[speaker].append({
                    'start': float(row['start']),
                    'end': float(row['end'])
                })

# Kama wahusika bado hawapo, tengeneza mhusika mmoja wa default
if not wahusika:
    print('Hakuweza kutambua wahusika - itatumia sauti moja kwa wote')
    wahusika['SPEAKER_00'] = [{'start': 0, 'end': 999}]

print(f'\nWahusika waliopatikana: {len(wahusika)}')
for mhusika, muda in wahusika.items():
    jumla_muda = sum(k['end'] - k['start'] for k in muda)
    print(f'  {mhusika}: mazungumzo {len(muda)}, jumla {jumla_muda:.1f} sekunde')

# ═══════════════════════════════════════════
# HATUA 2: Toa sampuli ya sauti kwa kila mhusika
# ═══════════════════════════════════════════
print('\n' + '='*50)
print('HATUA 2: Inachukua sampuli za wahusika...')
print('='*50)

for mhusika, muda_list in wahusika.items():
    kipande_bora = max(muda_list, key=lambda x: x['end'] - x['start'])
    mwanzo = kipande_bora['start']
    muda = min(kipande_bora['end'] - mwanzo, 20)

    faili_sampuli = os.path.join(FOLDA_SAMPULI, f'{mhusika}.wav')
    subprocess.run([
        FFMPEG,
        '-i', FILAMU,
        '-ss', str(mwanzo),
        '-t', str(muda),
        '-vn', '-ar', '22050',
        '-y', faili_sampuli
    ], capture_output=True)
    print(f'  Sampuli ya {mhusika}: {muda:.1f} sekunde')

# ═══════════════════════════════════════════
# HATUA 3: Pakia Coqui XTTS-v2
# ═══════════════════════════════════════════
print('\n' + '='*50)
print('HATUA 3: Inapakia Coqui XTTS-v2...')
print('Mara ya kwanza itapakua GB 2 - subiri...')
print('='*50)

tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)

# ═══════════════════════════════════════════
# HATUA 4: Soma tafsiri
# ═══════════════════════════════════════════
print('\n' + '='*50)
print('HATUA 4: Inasoma tafsiri...')
print('='*50)

with open(TAFSIRI, 'r', encoding='utf-8') as f:
    data = json.load(f)

def pata_mhusika(wakati_kuanza, wakati_kuisha):
    bora_mhusika = None
    bora_overlap = 0
    for mhusika, muda_list in wahusika.items():
        for kipande in muda_list:
            overlap_start = max(wakati_kuanza, kipande['start'])
            overlap_end = min(wakati_kuisha, kipande['end'])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > bora_overlap:
                bora_overlap = overlap
                bora_mhusika = mhusika
    return bora_mhusika or list(wahusika.keys())[0]

# ═══════════════════════════════════════════
# HATUA 5: Tengeneza sauti kwa kila sentensi
# ═══════════════════════════════════════════
print('\n' + '='*50)
print('HATUA 5: Inaunda sauti za Kiswahili...')
print('='*50)

jumla = len(data)
mafanikio = 0
makosa = 0

for i, kipande in enumerate(data):
    mhusika = pata_mhusika(kipande['start'], kipande['end'])
    sampuli = os.path.join(FOLDA_SAMPULI, f'{mhusika}.wav')
    faili_sauti = os.path.join(FOLDA_SAUTI, f'sauti_{i:04d}.wav')

    if not os.path.exists(sampuli):
        sampuli = SAMPULI_JUMLA

    maandishi = kipande['kiswahili'].strip()
    if not maandishi:
        print(f'[{i+1}/{jumla}] RUKA - maandishi yapo wazi')
        continue

    try:
        tts.tts_to_file(
            text=maandishi,
            speaker_wav=sampuli,
            language='en',
            file_path=faili_sauti
        )
        print(f'[{i+1}/{jumla}] ✓ {mhusika}: {maandishi[:45]}')
        mafanikio += 1
    except Exception as e:
        print(f'[{i+1}/{jumla}] ✗ KOSA: {e}')
        makosa += 1

print('\n' + '='*50)
print(f'IMEKAMILIKA!')
print(f'Sauti zilizofanikiwa: {mafanikio}/{jumla}')
print(f'Makosa: {makosa}')
print('='*50)