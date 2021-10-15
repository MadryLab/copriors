def rec_fn_apply(func, d):
    if isinstance(d, dict):
        return {k: rec_fn_apply(func, v) for k,v in d.items()}
    else:
        return func(d)

def to_numpy(arr):
    return arr.detach().cpu().numpy()