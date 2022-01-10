backslash_map = { '\a': r'\a', '\b': r'\b', '\f': r'\f',
                  '\n': r'\n', '\r': r'\r', '\t': r'\t', '\v': r'\v' }

def reconstruct_broken_string(s):
    for key, value in backslash_map.items():
        s = s.replace(key, value)
    return s