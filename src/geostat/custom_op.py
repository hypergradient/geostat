from .op import Op
from .param import get_parameter_values


__all__ = ['function']


class CustomOp(Op):
    def __init__(self, f, **fa):
        self.f = f
        super().__init__(fa, {})
    def __call__(self, e):
        v = get_parameter_values(self.fa)
        return self.f(*v.values())
    def vars(self):
        # Assume for now that all arguments are Parameters
        return {p.name: p for p in fa.values()}


def function(f):
    """
    A custom op involves two functions:
      * f, where it is defined, and
      * g, where it is introduced into the graph.
    """
    def g(*args):
      # Assume for now that all arguments are Parameters
      return CustomOp(f, **{p.name: p for p in args})
    return g
