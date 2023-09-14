import glob
import os
import re
import statistics as stat

inp = ["dsads", "wisdm"]
mem = [10, 20, 40]

for inp_ in inp:
    micro, macro = [], []
    for i in range(1, 6):
        if inp_ == 'dsads':
            path = f"eeil_{inp_}_{i}/har_flex_eeil/stdout*.txt"
        elif inp_ == 'wisdm':
            path = f"eeil_{inp_}_{i}/wisdm_flex_eeil/stdout*.txt"
        
        txt = glob.glob(path)[0]
        with open(txt, 'r') as f:
            for line in f.readlines():
                if line.strip().startswith("f1_score_micro"):
                    micro.append(float(line.strip().split(":")[1]))
                if line.strip().startswith("f1_score_macro"):
                    macro.append(float(line.strip().split(":")[1]))
    f1_micro_mean, f1_micro_std = stat.mean(micro) * 100, stat.pstdev(micro) * 100
    f1_macro_mean, f1_macro_std = stat.mean(macro) * 100, stat.pstdev(macro) * 100
    print(f"{inp_}_\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")
            