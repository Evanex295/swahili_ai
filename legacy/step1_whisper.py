import whisper
import json
import os

# Ongeza FFmpeg kwenye PATH
os.environ["PATH"] = r"C:\ffmpeg\ffmpeg-8.1-essentials_build\bin" + os.pathsep + os.environ["PATH"]

FILAMU = r'C:\swahili_ai_project\filamu_asili\Dominance.mp4'
MATOKEO = r'C:\swahili_ai_project\tafsiri\maandishi_asili.json'

print('Inapakia model ya Whisper...')
print('Subiri dakika 2-5 kwa mara ya kwanza...')

model = whisper.load_model('medium')
print('Model imepakiwa! Inaanza kusikia filamu...')

matokeo = model.transcribe(
    FILAMU,
    word_timestamps=True,
    verbose=True
)

os.makedirs(os.path.dirname(MATOKEO), exist_ok=True)
with open(MATOKEO, 'w', encoding='utf-8') as f:
    json.dump(matokeo, f, indent=2, ensure_ascii=False)

print(f'\nIMEMALIZA!')
print(f'Vipande vilivyopatikana: {len(matokeo["segments"])}')
print(f'Matokeo yamehifadhiwa: {MATOKEO}')