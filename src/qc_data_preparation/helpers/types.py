def lazy_dict(*args):
    return {f"{arg.replace('params:','')}": arg for arg in args}
