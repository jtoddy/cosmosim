# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:21:42 2022

@author: joeto
"""

import pstats
from pstats import SortKey

p_gen = pstats.Stats('restats_gen').strip_dirs()
p_gen_gpu = pstats.Stats('restats_gen_gpu').strip_dirs()

p_gen.sort_stats(SortKey.CUMULATIVE).print_stats(25)
print("-"*100)
p_gen_gpu.sort_stats(SortKey.CUMULATIVE).print_stats(25)