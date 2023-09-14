import glob
import os
import re
import statistics as stat

inp = ["dsads_e50_50", "pamap_e50_50", "hapt_e50_50", "wisdm_e50_50", "wisdm_e50_50_herd", "wisdm_e50_50_max", "wisdm_old_herd"]
mem = [10, 20, 40]

def mem_utils():
    for inp_ in inp:
        for mem_ in mem:
            micro, macro = [], []
            for i in range(1, 6):
                path = f"{inp_}_{mem_}_{i}.txt"
                with open(path, 'r') as f:
                    for line in f.readlines():
                        if line.strip().startswith("f1_score(micro)"):
                            micro.append(float(line.strip().split(":")[1]))
                        if line.strip().startswith("f1_score(macro)"):
                            macro.append(float(line.strip().split(":")[1]))
            f1_micro_mean, f1_micro_std, f1_micro_var = stat.mean(micro), stat.pstdev(micro), stat.pvariance(micro)
            f1_macro_mean, f1_macro_std, f1_macro_var = stat.mean(macro), stat.pstdev(macro), stat.pvariance(macro)
            print(f"{inp_}_{mem_}\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f}/{f1_micro_var:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f}/{f1_macro_var:.2f})")        
        print()    

def non_mem_utils():
    for inp_ in inp:
        micro, macro = [], []
        low, high = 1, 16
        if inp_ == "wisdm_old_herd":
            high = 6
        for i in range(low, high):
            path = f"{inp_}_{i}.txt"
            with open(path, 'r') as f:
                for line in f.readlines():
                    if line.strip().startswith("f1_score(micro)"):
                        micro.append(float(line.strip().split(":")[1]))
                    if line.strip().startswith("f1_score(macro)"):
                        macro.append(float(line.strip().split(":")[1]))
        
        # print(sorted(micro, reverse=True))
        # print(sorted(macro, reverse=True))

        top_5_micro = sorted(micro, reverse=True)[:5]
        top_5_macro = sorted(macro, reverse=True)[:5]

        # f1_micro_mean, f1_micro_std, f1_micro_var = stat.mean(micro), stat.pstdev(micro), stat.pvariance(micro)
        # f1_macro_mean, f1_macro_std, f1_macro_var = stat.mean(macro), stat.pstdev(macro), stat.pvariance(macro)
        # print(f"{inp_}_\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f}/{f1_micro_var:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f}/{f1_macro_var:.2f})")        
        f1_micro_mean, f1_micro_std, f1_micro_var = stat.mean(top_5_micro), stat.pstdev(top_5_micro), stat.pvariance(top_5_micro)
        f1_macro_mean, f1_macro_std, f1_macro_var = stat.mean(top_5_macro), stat.pstdev(top_5_macro), stat.pvariance(top_5_macro)        
        print(f"{inp_}_\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")        
    print()        

if __name__ == '__main__':
    # mem_utils()
    non_mem_utils()