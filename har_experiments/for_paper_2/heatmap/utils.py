import statistics as stat

from sklearn.metrics import f1_score


def find_min():
    inp = ["gdumb_dsads", "gdumb_pamap", "gdumb_hapt", "gdumb_wisdm"]
    for inp_ in inp:
        mic_min, mic_idx = 100, 0
        mics = []
        for i in range(1, 6):
            path = f"{inp_}_{i}.txt"
            with open(path) as f:
                lines = f.readlines()
                for l in lines:
                    if l.strip().startswith("f1_micro"):
                        mic = float(l.strip().split(":")[-1].strip())
                        mics.append(mic)
                        if mic_min > mic:
                            mic_min = mic
                            mic_idx = i
        print(f"Minimum f1-micro: {mic_min:.2f} found at index: {mic_idx}")
        # print(mics)

def aggregate():
    inp = ['ve_h_dsads', 've_h_pamap', 've_h_hapt', 've_h_wisdm',
            've_a_dsads', 've_a_pamap', 've_a_hapt', 've_a_wisdm',
            've_m_dsads', 've_m_pamap', 've_m_hapt', 've_m_wisdm']
    for inp_ in inp:
        mic, mac = [], []
        for i in range(1, 6):
            path = f"{inp_}_{i}.txt"
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("f1_micro"):
                        mic.append(float(line.split(":")[1]))
                    if line.startswith("f1_macro"):
                        mac.append(float(line.split(":")[1]))

        f1_micro_mean, f1_micro_std, f1_micro_var = stat.mean(mic), stat.pstdev(mic), stat.pvariance(mic)
        f1_macro_mean, f1_macro_std, f1_macro_var = stat.mean(mac), stat.pstdev(mac), stat.pvariance(mac)
        print(f"{inp_}\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f})")

aggregate()