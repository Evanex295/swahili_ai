import json
import os
import asyncio
import subprocess
import edge_tts

TAFSIRI = r'C:\swahili_ai_project\tafsiri\tafsiri_kiswahili.json'
FOLDA_SAUTI = r'C:\swahili_ai_project\sauti_vipande'
FFMPEG = r'C:\ffmpeg\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe'

SAUTI = "sw-TZ-DaudiNeural"  # Sauti ya kiume ya Tanzania
# SAUTI = "sw-TZ-RehemaNeural"  # Sauti ya kike ya Tanzania

async def tengeneza_sauti(maandishi, faili_wav):
    # edge_tts hutoa MP3 - tunahifadhi kama temp kisha tubadilishe kuwa WAV
    faili_mp3 = faili_wav.replace('.wav', '_temp.mp3')
    communicate = edge_tts.Communicate(maandishi, SAUTI)
    await communicate.save(faili_mp3)
    subprocess.run(
        [FFMPEG, '-i', faili_mp3, '-ar', '22050', '-ac', '1', '-y', faili_wav],
        capture_output=True, check=True
    )
    os.remove(faili_mp3)

async def main():
    with open(TAFSIRI, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(FOLDA_SAUTI, exist_ok=True)
    jumla = len(data)
    print(f'Kutengeneza sauti {jumla} za Kiswahili...')

    for i, kipande in enumerate(data):
        maandishi = kipande['kiswahili'].strip()
        if not maandishi:
            print(f'[{i+1}/{jumla}] RUKA - maandishi yapo wazi')
            continue
        faili_sauti = os.path.join(FOLDA_SAUTI, f'sauti_{i:04d}.wav')
        await tengeneza_sauti(maandishi, faili_sauti)
        print(f'[{i+1}/{jumla}] {maandishi[:50]}')

    print('\nIMEKAMILIKA! Sauti zote zimetengenezwa.')

asyncio.run(main())