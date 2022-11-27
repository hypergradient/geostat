from dataclasses import dataclass
from typing import Dict

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

@dataclass
class Op:
    """
    The `autoinputs` parameter contains a list of upstream ops,
    so it links ops together in a DAG. We walk the DAG for various
    reasons.

      - `run` calls the op and puts the output in `self.out`,
        after recursively doing this for autoinputs.
      - `gather_vars` recursively gathers variables from self and
        autoinputs.
    """
    fa: Dict[str, object] # Formal arguments.
    autoinputs: object # Blob of Ops.

    def vars(self): # Parameters
        pass

    def gather_vars(self, dejavu=None):
        if dejavu is None: dejavu = {}
        if id(self) not in dejavu:
            vv = {v for op in tf.nest.flatten(self.autoinputs) for v in op.gather_vars(dejavu)}
            dejavu[id(self)] = vv | set(self.vars())
        return dejavu[id(self)]

    def __call__(self, p, **e):
        pass

    def run(self, cache, p, **e):
        """
        If op has already been run, return result. Else:
            - Assemble inputs by recursively calling upstream ops.
            - Execute op.
            - Store result in self.out.
        """
        if id(self) not in cache:
            xval = tf.nest.map_structure(lambda op: op.run(cache, p, **e), self.autoinputs)
            cache[id(self)] = self(p, **e, auto=xval)
        return cache[id(self)]

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(type(self))

class SingletonTraceType(tf.types.experimental.TraceType):
    """
    A trace type to override TF's default behavior, which is
    to treat dataclass-based onjects as dicts.
    """

    def __init__(self, classtype):
       self.classtype = classtype
       pass

    def is_subtype_of(self, other):
       return type(other) is SingletonTraceType \
           and self.classtype is other.classtype

    def most_specific_common_supertype(self, others):
       return self if all(self == other for other in others) else None

    def __eq__(self, other):
       return isinstance(other, SingletonTraceType) and self.classtype == other.classtype

    def __hash__(self):
       return hash(self.classtype)
