## fixed seed setting
import random, torch, numpy as np
def random_ctl(use_seed=0):
    seed = use_seed if use_seed else random.randint(1,1000000)
    print(f"Using seed: {seed}")

    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if use_seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # numpy RNG
    np.random.seed(seed)

    return seed

if __name__ == "__main__":
    # set to 0 to be in random mode, or to a specific value if you want a specific reproducible case
    bad_seed = 0
    #bad_seed = 5079
    random_ctl(bad_seed)
