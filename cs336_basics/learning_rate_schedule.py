from math import cos, pi

def get_cos_lr(
    t: int, # current iteration
    alpha_max: float, # maximum learning rate
    alpha_min: float, # minimum (final) learning rate
    T_w: int, # number of warm-up iterations
    T_c: int, # final iteration of cosine annealing
) -> float:
    if t < T_w:
        return t / T_w * alpha_max
    elif t <= T_c:
        return alpha_min + (1 + cos((t - T_w) * pi / (T_c - T_w))) / 2 * (alpha_max - alpha_min)
    else:
        return alpha_min
