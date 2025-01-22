from dataclasses import dataclass
from tensorflow.types.experimental import TraceType
from typing import Dict

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .param import get_parameter_values

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
        """
        `cache` maps from Op ids to sets of variable names.

        Returns a dict of parameters, keyed by name.
        """
        if cache is None: cache = {}
        if id(self) not in cache:
            vv = {k: v for op in tf.nest.flatten(self.autoinputs)
                       if isinstance(op, Op)
                       for k, v in op.gather_vars(cache).items()}
            # print(self, '<-', [x.name for x in vv], '|', [x.name for x in set(self.vars())], '\n')
            cache[id(self)] = vv | self.vars()
        return cache[id(self)]

    def __call__(self, e):
        """
        `e` is a dict of params and evaluated inputs from upstream ops.
        Other values in `e` are supplied by the caller.
        """
        pass

    def run(self, cache):
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
                return op.run(cache)

        if id(self) not in cache:
            e = tf.nest.map_structure(eval, self.autoinputs)
            e |= get_parameter_values(self.fa)
            cache[id(self)] = self(e)

            # Save the Op so that its ID remains unique.
            if '__save__' not in cache: cache['__save__'] = []
            cache['__save__'].append(self)

        return cache[id(self)]

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(self)


class SingletonTraceType(TraceType):
    """
    A trace type to override TF's default behavior, which is
    to treat dataclass-based onjects as dicts.
    """
    def __init__(self, thing):
        self.value = thing

    def is_subtype_of(self, other):
        return self.value is other.value

    def most_specific_common_supertype(self, other):
        if self.value is other.value:
            return self.value
        else:
            return None

    def placeholder_value(self, placeholder_context):
        return self.value

    def __eq__(self, other):
        return self.value is other.value

    def __hash__(self):
        return hash(id(self.value))
