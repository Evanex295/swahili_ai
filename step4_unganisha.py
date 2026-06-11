import json
import os
import subprocess

TAFSIRI = r'C:\swahili_ai_project\tafsiri\tafsiri_kiswahili.json'
FOLDA_SAUTI = r'C:\swahili_ai_project\sauti_vipande'
FILAMU_ASILI = r'C:\swahili_ai_project\filamu_asili\Dominance.mp4'
FILAMU_MWISHO = r'C:\swahili_ai_project\matokeo\dominance_kiswahili.mp4'

FFMPEG = r'C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe'

with open(TAFSIRI, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Pata muda wa filamu asili
probe = subprocess.run([
    FFMPEG.replace('ffmpeg.exe', 'ffprobe.exe'),
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', FILAMU_ASILI
], capture_output=True, text=True)
muda_wote = float(probe.stdout.strip())

# Tengeneza sauti ya ukimya (silence)
silence = os.path.join(FOLDA_SAUTI, 'ukimya.wav')
subprocess.run([
    FFMPEG, '-f', 'lavfi', '-i',
    f'anullsrc=r=24000:cl=mono', '-t', str(muda_wote),
    '-y', silence
], check=True)

filter_parts2 = []
for i, kipande in enumerate(data):
    filter_parts2.append(
        f'[{i+2}]adelay={int(kipande["start"]*1000)}|{int(kipande["start"]*1000)}[s{i}]'
    )
mix2 = ''.join(f'[s{i}]' for i in range(len(data)))
filter_parts2.append(f'[1]{mix2}amix=inputs={len(data)+1}:normalize=0[out]')

subprocess.run([
    FFMPEG,
    '-i', FILAMU_ASILI,
    '-i', silence
] + [item for i in range(len(data)) for item in ['-i', os.path.join(FOLDA_SAUTI, f'sauti_{i:04d}.wav')]] + [
    '-filter_complex', ';'.join(filter_parts2),
    '-map', '0:v',
    '-map', '[out]',
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-shortest', '-y',
    FILAMU_MWISHO
], check=True)

print(f'\nFILAMU IMEKAMILIKA!')
print(f'Tazama: {FILAMU_MWISHO}')