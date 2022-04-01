# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:21:42 2022

@author: joeto
"""

import pstats
from pstats import SortKey

p_gen = pstats.Stats('restats_gen').strip_dirs()
p_gen_new = pstats.Stats('restats_gen_new').strip_dirs()
p_play = pstats.Stats('restats_play').strip_dirs()

p_gen.sort_stats(SortKey.CUMULATIVE).print_stats(25)
print("-"*100)
p_gen_new.sort_stats(SortKey.CUMULATIVE).print_stats(25)
print("-"*100)
p_play.sort_stats(SortKey.CUMULATIVE).print_stats(25)