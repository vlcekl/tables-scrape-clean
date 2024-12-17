import random
import numpy as np
from numpy import cumprod
from numpy.random import binomial

budget = 1000

prob = 0.6

nsim = 10000

nbets = 1000

bet = 0.2

for i in range(100):
    bet = 0.01 * i
    gains = [cumprod( 1 + bet * (2*binomial(1, prob, size=nbets) - 1))[1] for _ in range(nsim)]
    gain_ave = np.mean(gains)
    gain_sd = np.std(gains)
    gain_min = np.min(gains)
    print(bet, round(gain_ave, 3), round(gain_sd, 3), round(gain_min, 3))