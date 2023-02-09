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
    The `autoinputs` parameter contains a blob of upstream ops. The leaves
    in the blob are either the op itself or a string identifier. In the
    latter case, the string identifier should be present as a key in the
    `cache` that gets passed in.

    This parameter links ops together in a DAG. We walk the DAG for
    various reasons.

      - `run` calls the op and puts the output in `self.out`,
        after recursively doing this for autoinputs.
      - `gather_vars` recursively gathers variables from self and
        autoinputs.
    """
    fa: Dict[str, object] # Formal arguments.
    autoinputs: object # Blob of Ops and strings.

    def vars(self): # Parameters
        return []

    def gather_vars(self, cache=None):
        if cache is None: cache = {}
        if id(self) not in cache:
            vv = {v for op in tf.nest.flatten(self.autoinputs) if isinstance(op, Op) for v in op.gather_vars(cache)}
            cache[id(self)] = vv | set(self.vars())
        return cache[id(self)]

    def __call__(self, p, e):
        """
        `p` is a dict of model parameters.

        `e` is a blob of evaluated inputs from upstream ops and other
        things inserted by the caller.  Other values in `e` are supplied
        by the caller.
        """
        pass

    def run(self, cache, p):
        """
        If op has already been run, return result. Else:
            - Assemble inputs by recursively calling upstream ops.
            - Execute op by calling `__call__`.
            - Store result in cache.
        """

        def eval(op):
            """
            Evaluate `op`. If `op` is a string, look up its value.
            Otherwise execute it.
            """
            if isinstance(op, str):
                return cache[op]
            else:
                return op.run(cache, p)
       
        if id(self) not in cache:
            e = tf.nest.map_structure(lambda op: eval(op), self.autoinputs)
            cache[id(self)] = self(p, e)

            # Save the Op so that its ID remains unique.
            if '__save__' not in cache: cache['__save__'] = []
            cache['__save__'].append(self)

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
