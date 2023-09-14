import statistics as stat

inp = ["dsads1_4", "pamap1_4", "hapt1_4"]
inp = ["dsads", "wisdm"]
mem = [10, 20, 40]

# f1_micro: 0.8433205057724025
# f1_macro: 0.8494044252188694

def aggregate():
    for inp_ in inp:
        mic, mac = [], []
        for i in range(1, 6):            
            in_ = f"gdumb_{inp_}_{i}.txt"
            with open(in_) as f:
                lines = f.readlines()
                for l in lines:
                    if l.strip().startswith("f1_micro"):
                        mic.append(float(l.strip().split(":")[-1].strip()))
                    if l.strip().startswith("f1_macro"):
                        mac.append(float(l.strip().split(":")[-1].strip()))
        f1_micro_mean, f1_micro_std = stat.mean(mic), stat.pstdev(mic)
        f1_macro_mean, f1_macro_std = stat.mean(mac), stat.pstdev(mac)
        print(f"{inp_}_\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")

aggregate()