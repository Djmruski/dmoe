import glob
import statistics as stat

inp = ["dsads1_4", "pamap1_4", "hapt1_4", "wisdm"]
inp = ['dsads', 'dsads_max', 'joint_wisdm', 'finetune_wisdm', 'wisdm', 'wisdm_max', 'gdumb_dsads', 'gdumb_wisdm']
mem = [10]

# f1_micro: 0.8433205057724025
# f1_macro: 0.8494044252188694

def aggregate():
    for inp_ in inp:
        for mem_ in mem:
            mic, mac = [], []
            for i in range(1, 6):            
                in_ = f"{inp_}_{mem_}_{i}.txt"
                with open(in_) as f:
                    lines = f.readlines()
                    for l in lines:
                        if l.strip().startswith("f1_micro"):
                            mic.append(float(l.strip().split(":")[-1].strip()))
                        if l.strip().startswith("f1_macro"):
                            mac.append(float(l.strip().split(":")[-1].strip()))
            f1_micro_mean, f1_micro_std = stat.mean(mic), stat.pstdev(mic)
            f1_macro_mean, f1_macro_std = stat.mean(mac), stat.pstdev(mac)
            print(f"{inp_}_{mem_}\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")

def aggregate20():
    inp = ['20_hapt']
    mem = [10]

    for inp_ in inp:
        for mem_ in mem:
            mic, mac = [], []
            for i in range(1, 21):            
                in_ = f"{inp_}_{mem_}_{i}.txt"
                with open(in_) as f:
                    lines = f.readlines()
                    for l in lines:
                        if l.strip().startswith("f1_micro"):
                            mic.append(float(l.strip().split(":")[-1].strip()))
                        if l.strip().startswith("f1_macro"):
                            mac.append(float(l.strip().split(":")[-1].strip()))
            mic = sorted(mic)[:5]
            mac = sorted(mac)[:5]
            
            f1_micro_mean, f1_micro_std = stat.mean(mic), stat.pstdev(mic)
            f1_macro_mean, f1_macro_std = stat.mean(mac), stat.pstdev(mac)
            print(f"{inp_}_{mem_}\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")

def aggregte_flex():
    # incl = [1, 2, 5] # for dsads_max
    for inp_ in inp:
        mic, mac = [], []
        for i in range(1, 6):
        # for i in incl:
            in_ = f"{inp_}_{i}.txt"
            with open(in_) as f:
                lines = f.readlines()
                for l in lines:
                    if l.strip().startswith("f1_score(micro)"):
                        mic.append(float(l.strip().split(":")[-1].strip()))
                    if l.strip().startswith("f1_micro"):
                        mic.append(100 * float(l.strip().split(":")[-1].strip()))                        
                    if l.strip().startswith("f1_score(macro)"):
                        mac.append(float(l.strip().split(":")[-1].strip()))
                    if l.strip().startswith("f1_macro"):
                        mac.append(100 * float(l.strip().split(":")[-1].strip()))                        
        
        f1_micro_mean, f1_micro_std = stat.mean(mic), stat.pstdev(mic)
        f1_macro_mean, f1_macro_std = stat.mean(mac), stat.pstdev(mac)
        print(f"{inp_}\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")

def aggregate_flex_facil():
    inp = ['eeil', 'icarl']
    for inp_ in inp:
        micro, macro = [], []
        for i in range(1, 6):
            if inp_ == 'eeil':
                path = f"eeil_e5_wisdm_{i}/wisdm_flex_eeil/stdout*.txt"                          
            else:
                path = f"icarl_wisdm_{i}/wisdm_flex_icarl/stdout*.txt"                          
            
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
                

prefix = [f"{i}%" for i in range(0, 101)]

def clean_tqdm(inp):
    for inp_ in inp:
        for i in range(1, 6):
            out = []
            in_ = f"{inp_}_{i}.txt"
            with open(in_) as f:
                lines = f.readlines()
                for l in lines:
                    if not l.strip().startswith(tuple(prefix)):
                        out.append(l)
                
            with open(in_, "w") as f:
                f.write(''.join(out))        

# aggregate()
# aggregate20()
# clean_tqdm(['joint_wisdm', 'finetune_wisdm'])
aggregte_flex()
# aggregate_flex_facil()