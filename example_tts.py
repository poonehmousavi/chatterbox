import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import pickle
# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
emo_dir_path= 'src/all_emo_dirs.pkl'
emo_strategy='predefined'
emotion='happy'
model = ChatterboxTTS.from_pretrained(device=device)
model.set_emo_dirs(emo_dir_path)
# print(model.emo_dirs.keys())

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."


exaggeration =0.5
cfg_weight =0.5
# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "C:/Users/pooneh.mousavi/projects/chatterbox/000_Frank Seagrass.wav"
# Correct parameter name: emotion_strength (was misspelled in older code as emotion_stength)
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH,exaggeration=exaggeration,cfg_weight=cfg_weight)
# ta.save(f"Frank_Seagrass/ex{exaggeration}-cf-{cfg_weight}/TTS_Frank_Seagrass_sythesized{exaggeration}-{cfg_weight}.wav", wav, model.sr)


# # audio_pairs = [
#     ('C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/original_0.wav', 'C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/angry_0.wav'),
#     ('C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/original_1.wav', 'C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/angry_1.wav')
# ]
# 
emotion_strength =0.8 
emotion_strategy ="predefined"
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH,exaggeration=exaggeration,cfg_weight=cfg_weight, emotion_strength=emotion_strength, emotion=emotion, emotion_strategy=emo_strategy)
# ta.save(f"frank_seagrass_syth/frank_seagrass_predefined_emo_{emotion}{exaggeration}-{cfg_weight}-{emotion_strength}.wav", wav, model.sr)
# emotions =[ "angry", "blaming", "charisma", "contempt",'curious, intrigued', 'desire and excitement', 'desire', "empathy", 'envy', 'grateful, appreciative, thankful, indebted, blessed',"happy","romance","sad",'sarcasm',"surprise"]
emotions=['charisma', 'empathetic', 'angry', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'desire', 'doubt', 'empathic pain', 'envy', 'joy', 'neutral', 'romance', 'sarcasm', 'tiredness', 'triump']
# emotion = "empathy"
for emotion in emotions:    
    audio_pairs = [
        ('C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/original_0.wav', f'C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/{emotion}_0.wav'),
        ('C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/original_1.wav', f'C:/Users/pooneh.mousavi/projects/chatterbox/emotions_audio/{emotion}_1.wav')
    ]

      
    # AUDIO_PROMPT_PATH = "000_Frank Seagrass.wav"
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH,exaggeration=exaggeration,cfg_weight=cfg_weight, emotion_strategy=emotion_strategy,emotion_strength=emotion_strength, emotion_desc=emotion)
    ta.save(f"Frank_Seagrass/ex{exaggeration}-cf-{cfg_weight}-predefined/TTS_Frank_Seagrass_sythesized_emo_{emotion}{exaggeration}-{cfg_weight}-{emotion_strength}.wav", wav, model.sr)