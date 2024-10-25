from melo.api import TTS
# from IPython.display import Audio

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "auto"  # Will automatically use GPU if available
speed = 1.0

# English
# text = "Did you ever hear a folk tale about a giant turtle?"
print("hello texting on ")
model = TTS(language="EN", device=device)
speaker_ids = model.hps.data.spk2id

text = "In the heart of the bustling city, where the sounds of traffic blend with the laughter of children playing in the park, a small café stands quietly, inviting passersby with the rich aroma of freshly brewed coffee and baked pastries; it is a place where stories are shared, friendships are formed, and time seems to slow down just enough for one to savor the simple pleasures of life"
output_path = "./cpu_text_en-us.wav"
model.tts_to_file(text, speaker_ids["EN-Default"], output_path, speed=speed)
