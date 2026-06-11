import json
import time
import os
from groq import Groq

MAANDISHI_ASILI = r'C:\swahili_ai_project\tafsiri\maandishi_asili.json'
TAFSIRI_KISWAHILI = r'C:\swahili_ai_project\tafsiri\tafsiri_kiswahili.json'

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def tafsiri_sentensi(sentensi, nambari, jumla):
    jibu = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Wewe ni mtafsiri bora wa filamu.
Tafsiri sentensi hii kuwa Kiswahili cha kawaida cha Tanzania.

Kanuni MUHIMU:
1. Tumia Kiswahili cha mazungumzo ya kila siku - si cha vitabuni
2. Idadi ya maneno iwe KARIBU SAWA na ya asili
3. Hisia zibaki: hasira = hasira, furaha = furaha
4. Jibu na SENTENSI TU - bila maelezo mengine

Sentensi: {sentensi}

Tafsiri:"""
        }],
        max_tokens=300
    )
    
    kiswahili = jibu.choices[0].message.content.strip()
    print(f'[{nambari}/{jumla}] {sentensi[:40]}')
    print(f'         → {kiswahili}')
    return kiswahili

# Pakia maandishi asili
with open(MAANDISHI_ASILI, 'r', encoding='utf-8') as f:
    data = json.load(f)

vipande = data['segments']
jumla = len(vipande)
print(f'Sentensi za kutafsiri: {jumla}')

# Tafsiri kila sentensi
matokeo = []
for i, kipande in enumerate(vipande):
    kiswahili = tafsiri_sentensi(kipande['text'], i+1, jumla)
    matokeo.append({
        'start': kipande['start'],
        'end': kipande['end'],
        'kiingereza': kipande['text'],
        'kiswahili': kiswahili
    })
    time.sleep(0.2)

# Hifadhi matokeo
with open(TAFSIRI_KISWAHILI, 'w', encoding='utf-8') as f:
    json.dump(matokeo, f, indent=2, ensure_ascii=False)

print(f'\nIMEKAMILIKA! Tafsiri {jumla} zimehifadhiwa.')