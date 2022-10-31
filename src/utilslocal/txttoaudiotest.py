# %%
from gtts import gTTS
import os
import os
import time



# define variables
s = "I see 3 people. So you all can have 3 pieces of candy. "
file = "./file.mp3"
if os.path.exists(file):
    os.unlink(file)

# initialize tts, create mp3 and play
tts = gTTS(s)
tts.save(file)
# tts.
time.sleep(0.5)
# %%

# os.system("mpg123 " + file)

from pygame import mixer
mixer.init()
mixer.music.load(file)
mixer.music.play()
time.sleep(6)

# mixer.quit()

# time.sleep(5)
# mixer.music.stop()
# %%
