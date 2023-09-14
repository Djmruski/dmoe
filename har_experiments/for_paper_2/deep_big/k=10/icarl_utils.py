import glob
import statistics as stat

inp = ['dsads', 'wisdm']

def aggregate():
    for inp_ in inp:
        mic, mac = [], []
        for i in range(1, 6):            
            if inp_ == 'dsads':
                path = f"icarl_dsads_{i}/har_flex_icarl/stdout*.txt"                          
            elif inp_ == 'wisdm':
                path = f"icarl_wisdm_{i}/wisdm_flex_icarl/stdout*.txt"                          

            txt = glob.glob(path)[0]
            with open(txt, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if l.strip().startswith("f1_score_micro"):
                        mic.append(float(l.strip().split(":")[-1].strip()))
                    if l.strip().startswith("f1_score_macro"):
                        mac.append(float(l.strip().split(":")[-1].strip()))
        f1_micro_mean, f1_micro_std = 100 * stat.mean(mic), 100 * stat.pstdev(mic)
        f1_macro_mean, f1_macro_std = 100 * stat.mean(mac), 100 * stat.pstdev(mac)
        print(f"{inp_}_\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")

aggregate()