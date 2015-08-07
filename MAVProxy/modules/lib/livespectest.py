import livespec
from math import *
import time
import random

s = livespec.LiveSpectrogram(50, .5, .1, maxscale=1.0)

f = 0.0
phi = 0.0
dt = .02
while True:
    # RANDOM DATA
    time.sleep(dt)
    phi += f*2*pi*dt
    f += dt*.25
    s.new_sample(sin(phi)+random.gauss(0,1), time.time())
    if f > 25:
        f = 0.0
    if not s.is_alive():
        break


s.close()
