from dataclasses import dataclass

@dataclass
class PaperParameter:
    name: str
    lo: float
    hi: float

def get_parameter_values(
    blob: object,
    p: Dict[str, object]
):
    """
    For each string encountered in the nested blob,
    look it up in `p` and replace it with the lookup result.
    """
    if isinstance(blob, dict):
        return {k: get_parameter_values(a, p) for k, a in blob.items()}
    elif isinstance(blob, (list, tuple)):
        return [get_parameter_values(a, p) for a in blob]
    elif isinstance(blob, str):
        if blob not in p:
            raise ValueError('Parameter `%s` not found' % blob)
        return p[blob]
    elif blob is None:
        return None
    else:
        return blob

def ppp(name):
    """Positive paper parameter (maybe)."""
    if isinstance(name, str):
        return [PaperParameter(name, 0., float('inf'))]
    else:
        return []

def upp(name):
    """Unbounded paper parameter (maybe)."""
    if isinstance(name, str):
        return [PaperParameter(name, float('-inf'), float('inf'))]
    else:
        return []

def bpp(name, lo, hi):
    """Bounded paper parameter (maybe)."""
    if isinstance(name, str):
        return [PaperParameter(name, lo, hi)]
    else:
        return []
