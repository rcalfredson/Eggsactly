import operator

# - - - tuples

# returns t2 as tuple
#  if t2 is int, float, or string, t2 is replicated len(t1) times
#  otherwise, t2 is passed through
def _toTuple(t1, t2):
  if isinstance(t2, (int,float,str)):
    t2 = len(t1) * (t2,)
  return t2

# applies the given operation to the given tuples; t2 can also be number
def tupleOp(op, t1, t2):
  return tuple(map(op, t1, _toTuple(t1, t2)))

# tupleOp() add
def tupleAdd(t1, t2): return tupleOp(operator.add, t1, t2)

# tupleOp() subtract
def tupleSub(t1, t2): return tupleOp(operator.sub, t1, t2)

# tupleOp() multiply
def tupleMul(t1, t2): return tupleOp(operator.mul, t1, t2)