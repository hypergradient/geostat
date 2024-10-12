from dataclasses import dataclass
from typing import Dict

from jax.tree_util import tree_flatten
from jax.tree_util import tree_map

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
        if cache is None:
            cache = {}
        if id(self) not in cache:
            vv = {k: v for op in tree_flatten(self.autoinputs)[0]  # Use tree_flatten for handling nested structures
                if isinstance(op, Op)
                for k, v in op.gather_vars(cache).items()}
            cache[id(self)] = {**vv, **self.vars()}
        return cache[id(self)]

    def __call__(self, e):
        """
        `e` is a dict of params and evaluated inputs from upstream ops.
        Other values in `e` are supplied by the caller.
        """
        pass

    def run(self, cache):
        """
        If the operation has already been run, return the result. Otherwise:
            - Assemble inputs by recursively calling upstream operations.
            - Execute the operation by calling `__call__`.
            - Store the result in the cache.
        """

        def eval(op):
            """
            Evaluate `op`. If `op` is a string, look up its value.
            Otherwise, execute it.
            """
            if isinstance(op, str):
                return cache[op]
            else:
                return op.run(cache)

        if id(self) not in cache:
            # Assemble inputs by recursively calling upstream operations
            e = tree_map(lambda op: eval(op), self.autoinputs)
            e |= get_parameter_values(self.fa)
            
            # Execute the operation and store the result in the cache
            cache[id(self)] = self(e)

            # Save the operation to ensure its ID remains unique
            if '__save__' not in cache:
                cache['__save__'] = []
            cache['__save__'].append(self)

        return cache[id(self)]
