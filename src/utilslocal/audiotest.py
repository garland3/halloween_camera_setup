# https://www.youtube.com/watch?v=mD0eZBzAS7c
# https://s8.converto.io/en52/download-redirect/?id=O5v1ayaja5qbTZumdMbsT2RKG6qJiJcH

# %%
from pathlib import Path
import time
# %%
get_audio = False
audiodir = Path("audio")
audiodir.mkdir(exist_ok = True)

filename = audiodir / f"scarysounds.mp3"

if get_audio:
    import moviepy.editor as mp
    my_clip = mp.VideoFileClip(r"C:\Users\garla\Downloads\Scary_Halloween_Sounds_🎃_Unlimited_Royalty_Free_Sound_Effects.mp3")

    my_clip.audio.write_audiofile(filename)
else:
    from pygame import mixer  # Load the popular external library

    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()
    
    
    time.sleep(5)
    mixer.music.stop()
# %%
# %%


# import torchaudio
# from speechbrain.pretrained import Tacotron2
# from speechbrain.pretrained import HIFIGAN

# # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
# tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
# hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# # Running the TTS
# mel_output, mel_length, alignment = tacotron2.encode_text("Only take 1 candy if you dare. ")

# # Running Vocoder (spectrogram-to-waveform)
# waveforms = hifi_gan.decode_batch(mel_output)

# # Save the waverform
# torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050)

# from espnet2.bin.tts_inference import Text2Speech
# from espnet2.utils.types import str_or_none
# import time
# import torch

# # Other languages are support like Japanese, Chinese Mandarin and many others....
# lang = 'English'

# # Currently utilzing the Pre-trained VITs LJSpeech model, but can easily replace...
# tag = 'kan-bayashi/ljspeech_vits'

# # Replace the tag with any of the following acoustic models below
# # kan-bayashi/ljspeech_tacotron2, kan-bayashi/ljspeech_fastspeech ,
# # kan-bayashi/ljspeech_fastspeech2, kan-bayashi/ljspeech_conformer_fastspeech2, 
# # kan-bayashi/ljspeech_vits

# # Since VITS doesnt require a vocoder we arent utilizing one.
# # That doesnt stop you from using one though and experimenting.
# vocoder_tag = "none"

# Replace the tag with any of the following vocoders 
# parallel_wavegan/ljspeech_parallel_wavegan.v1, parallel_wavegan/ljspeech_full_band_melgan.v2
# parallel_wavegan/ljspeech_multi_band_melgan.v2, parallel_wavegan/ljspeech_hifigan.v1 ,
# parallel_wavegan/ljspeech_style_melgan.v1

# text2speech = Text2Speech.from_pretrained(
# model_tag=str_or_none(tag),
# vocoder_tag=str_or_none(vocoder_tag),
# device="cuda",

# # Only for Tacotron 2 & Transformer
# threshold=0.5,

# # Only for Tacotron 2
# minlenratio=0.0,
# maxlenratio=10.0,
# use_att_constraint=False,
# backward_window=1,
# forward_window=3,

# # Only for FastSpeech & FastSpeech2 & VITS
# speed_control_alpha=1.0,

# # Only for VITS
# noise_scale=0.333,
# noise_scale_dur=0.333,
# )

# # speech synthesis
# input_text = 'This is what you have been waiting for!'

# with torch.no_grad():
# start = time.time()
# wav = text2speech(input_text)["wav"]
# rtf = (time.time() - start) / (len(wav) / text2speech.fs)
# print(f"RTF = {rtf:5f}")

# # let us listen to generated samples
# from IPython.display import display, Audio
# display(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs))


# %%

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd


models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(model, cfg)

text = "Hello, this is a test run."

sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

ipd.Audio(wav, rate=rate)


# %%
