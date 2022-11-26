from dataclasses import dataclass
from typing import Dict

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

@dataclass
class Op:
    fa: Dict[str, object] # Formal arguments.

    def vars(self): # Parameters
        pass

    def run(self, x, p):
        pass

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
