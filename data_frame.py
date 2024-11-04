from typing import List, Any, Optional
import abc
from collections.abc import Iterable, Sized
from pprint import pformat
from tabulate import tabulate

def is_numeric(dtype):
    return dtype in (int, float)

class NoneHandler(abc.ABC):
    @abc.abstractmethod
    def can_output_optional(self):
        pass

    @abc.abstractmethod
    def wrap(self, op):
        pass

class PropagateNoneHandler(NoneHandler):
    def can_output_optional(self):
        return True

    def wrap(self, op):
        def wrapped(*args):
            for i in args:
                if i is None:
                    return None
            return op(*args)
        return wrapped

class RaiseNoneHandler(NoneHandler):
    def can_output_optional(self):
        return False
    
    def wrap(self, op):
        def wrapped(*args):
            try:
                out = op(*args)
            except TypeError as e:
                if "NoneType" in str(e):
                    raise ValueError("Operation returned None")
                else:
                    raise e
            if out is None:
                raise ValueError("Operation returned None")
            return out
        return wrapped
    
class ReplaceOutputNoneHandler(NoneHandler):
    def __init__(self, value):
        self.value = value

    def can_output_optional(self):
        return False
    
    def wrap(self, op):
        def wrapped(*args):
            try:
                out = op(*args)
            except TypeError as e:
                if "NoneType" in str(e):
                    return self.value
                else:
                    raise e
            if out is None:
                return self.value
            return out
        return wrapped

# a mask is a series of boolean values that can be used to filter a series
# a boolean series is a series of boolean or none values

# Left join
# 


class Series:
    comparison_none_handler = ReplaceOutputNoneHandler(False)
    others_none_handler = PropagateNoneHandler()
    def __init__(self, data: List[Any], dtype, is_optional, name: str = None):
        # Initializes a new Series object to enhance performance by using more efficient data structures.
        # By opting for arrays over Python lists, we avoid the overhead of handling a list of pointers to objects.
        # Arrays enable more compact data storage, which improves both space and time efficiencies. Specifically,
        # for a non-optional boolean array, utilizing a bit vector can be beneficial. This approach significantly 
        # speeds up elementwise operations and reduces memory usage.
        self.data = data
        self.name = name
        self.dtype = dtype
        self.optional = is_optional

    @classmethod
    def from_array_like(cls, data, name=None) -> "Series":
        # Creates a Series object from data that supports array-like operations. This method offers optional 
        # performance optimizations by allowing the user to control whether to make a copy of the data and 
        # to verify nullable values.

        # Although these options provide some degree of performance tuning, the overall impact may be limited.
        # Typically, when data is read from a file, making a copy and checking for null values are necessary steps 
        # to ensure data integrity and stability.
        if not (isinstance(data, Iterable) and isinstance(data, Sized)):
            raise ValueError("Data must be array-like")
        dtype = type(None)
        is_optional = False
        length = len(data)
        # we need some smart way to detect the type of the data, this way, we might have a problem when
        # input is [1, 2, 3.0]
        storage = [None] * length
        for i, v in enumerate(data):
            if v is None:
                is_optional = True
                storage[i] = None
                continue
            if dtype == type(None):
                dtype = type(v)
            if dtype != type(v):
                raise ValueError("All values must have the same type")
            storage[i] = v
        return Series(storage, dtype, is_optional, name)

    def is_numeric(self):
        return self.dtype in (int, float)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, idx):
        # in pandas, indexing is done differently, location selection is done using iloc
        # we could implement slice selection here
        if isinstance(idx, int):
            return self.data[idx]
        elif isinstance(idx, Series) and idx.dtype == bool:
            return self.apply_mask(idx)
        else:
            raise NotImplemented

    def __str__(self):
        name = self.name if self.name else "unamed"
        optional_str = "optional " if self.optional else ""
        type_str = optional_str + self.dtype.__name__
        return f"""Series({name} <{type_str}>, {pformat(self.data)})"""
    
    def apply_mask(self, mask: "Series") -> "Series":
        # We can improve by returning a view of the original series
        # Applies a boolean mask to the Series, returning a view of the original Series that includes only the 
        # elements where the mask is True. This method optimizes performance by creating a lazy view rather than 
        # a deep copy of the original Series.

        # A view in this context is essentially a reference to the original Series with additional metadata:
        # 1. Information on which elements are masked.
        # 2. The original values of the elements.
        # 3. An optional cache to store the results when the view is accessed multiple times.

        # The view supports:
        # - Writing: Conditionally updating values directly on the original Series based on the mask.
        # - Reading: When a value from the view is accessed, the mask is applied to the original Series to create 
        #   and possibly cache a new, concrete Series. This ensures that subsequent reads are efficient, with 
        #   contiguous memory access. (some trade-offs can be made here)
        if not isinstance(mask, Series):
            return ValueError("Mask must be a boolean series")
        if mask.dtype != bool:
            return ValueError("Mask must be a boolean series")
        if len(mask) != len(self):
            raise ValueError("Mask must have the same length as the series")
        if mask.optional:
            raise ValueError("Mask should be a non-optional boolean series")
        output = []
        for b, v in zip(mask, self):
            if b:
                output.append(v)
        return Series(output, self.dtype, is_optional=self.optional,  name=None)

    # these operations assume type checking is done before
    def _unop(self, op):
        # here the unop and biop should be both optimized regarding the data structure
        #  Optimization Strategies:
        # - **Inlining Lambda Operations**: Inline lambda functions within the execution flow to 
        #   eliminate function call overhead. This step ensures that nested function calls are minimized.
        # - **Precompile operators**: If operating within a fixed set of operations, generate optimized implementations
        #   for these operators and dispatch them based on the underlying data structure. We could apply SIMD and multi-threading
        #   to speed up the execution, dispatch appropriate operators based on the data type and operation type.
        # - **Data Packing**: For non-optional series, we could pack numeric values compactly and pack boolean values as bit 
        #   vectors to further reduce memory usage and increase operation speed.
        # - **Handling Optional Values**: For series with optional values, implement a bitmask to identify `None` 
        #   values efficiently, allowing SIMD operations to be applied directly to non-None values. 
        # - **Just-In-Time Compilation**: For supporting lazy evaluation and operation chaining, leverage JIT 
        #   compilation to dynamically compile and optimize new operators as needed.
        # - Some simple first step trial can be using numpy array and numba jit to optimize the operations

        output_storage = [None] * len(self)
        for i, v in enumerate(self):
            output_storage[i] = op(v)
        return output_storage
    
    def _biop(self, other, op):
        output_storage = [None] * len(self)
        if len(self) != len(other):
            raise ValueError("Series must have the same length")
        for i, (v1, v2) in enumerate(zip(self, other)):
            output_storage[i] = op(v1, v2)
        return output_storage

    def _other_dtype(self, other):
        return other.dtype if isinstance(other, Series) else type(other)

    def _execute(self, other, op, operation_type):
        # Optimization Strategy:
        # - **Lazy Evaluation**: When a chain of operations, such as `series_a * 3 + 5`, is performed,
        #   it's typical to compute each operation sequentially, requiring multiple passes over the data.
        #   However, utilizing the distributive property of fmap over arithmetic operations, we can combine
        #   these into a single pass. This is done by composing the operations into one lambda, e.g., 
        #   `lambda x: x * 3 + 5`, which is then applied directly to the series.
        # - **Just-In-Time Compilation**: To efficiently handle this single composed operation, JIT compilation 
        #   may be utilized to optimize the execution of the new, combined operator.
        # - **Evaluation Strategies**: For boolean operations, evaluation can be performed eagerly as these are
        #   typically less computationally intensive. For numeric operations, a lazy evaluation strategy is preferred
        #   to better leverage computational resources and optimize performance.
        other_dtype = self._other_dtype(other)
        
        # urr, python doesn't support type algebra well, made it a bit verbose
        if operation_type == "numeric_comparison":
            none_handler = self.comparison_none_handler
            output_type = bool
            if not (self.is_numeric() and is_numeric(other_dtype)):
                raise ValueError("Can only compare between numeric values or series")
        elif operation_type == "equality_comparison":
            none_handler = self.comparison_none_handler
            output_type = bool
            both_numeric = self.is_numeric() and is_numeric(other_dtype)
            if not (both_numeric or self.dtype == other_dtype):
                raise ValueError("Can only compare same type series or values")
        elif operation_type == "arithmetic":
            none_handler = self.others_none_handler
            output_type = float if float in (self.dtype, other_dtype) else int
            if not (self.is_numeric() and is_numeric(other_dtype)):
                raise ValueError("Can only perform arithmetic operations on numeric series or values")
        elif operation_type == "bool_op":
            none_handler = self.others_none_handler
            output_type = bool
            if not (self.dtype == bool and other_dtype == bool):
                raise ValueError("Can only perform boolean operations on boolean series")
        else:
            raise ValueError("Invalid operation type")
        other_is_optional = other.optional if isinstance(other, Series) else False
        is_optional = self.optional or other_is_optional
        is_optional = none_handler.can_output_optional() and is_optional

        if isinstance(other, Series):
            op = none_handler.wrap(op)
            output_storage = self._biop(other, op)
        else:
            uop = lambda a: op(a, other)
            wrapped_op = none_handler.wrap(uop)
            output_storage = self._unop(wrapped_op)
        return Series(output_storage, output_type, is_optional=is_optional)

    # numeric comparison operations
    def __rlt__(self, other):
        return self._execute(other, lambda a, b: b < a, "numeric_comparison")

    def __rle__(self, other):
        return self._execute(other, lambda a, b: b <= a, "numeric_comparison")
    
    def __rgt__(self, other):
        return self._execute(other, lambda a, b: b > a, "numeric_comparison")
    
    def __rge__(self, other):
         return self._execute(other, lambda a, b: b > a, "numeric_comparison")

    def __lt__(self, other):
        return self._execute(other, lambda a, b: a < b, "numeric_comparison")
    
    def __le__(self, other):
        return self._execute(other, lambda a, b: a <= b, "numeric_comparison")
    
    def __gt__(self, other):
        return self._execute(other, lambda a, b: a > b, "numeric_comparison")

    def __ge__(self, other):
        return self._execute(other, lambda a, b: a >= b, "numeric_comparison")

    # equality comparison
    def __eq__(self, other):
        return self._execute(other, lambda a, b: a == b, "equality_comparison")
        
    def __ne__(self, other):
        return self._execute(other, lambda a, b: a != b, "equality_comparison")

    # arithmetic operations
    def __add__(self, other):
        return self._execute(other, lambda a, b: a + b, "arithmetic")
    
    def __radd__(self, other):
        return self._execute(other, lambda a, b: b + a, "arithmetic")
    
    def __sub__(self, other):
        return self._execute(other, lambda a, b: a - b, "arithmetic")
    
    def __rsub__(self, other):
        return self._execute(other, lambda a, b: b - a, "arithmetic")
    
    def __mul__(self, other):
        return self._execute(other, lambda a, b: a * b, "arithmetic")

    def __rmul__(self, other):
        return self._execute(other, lambda a, b: b * a, "arithmetic")
    
    def __truediv__(self, other):
        return self._execute(other, lambda a, b: a / b, "arithmetic")
    
    def __rtruediv__(self, other):
        return self._execute(other, lambda a, b: b / a, "arithmetic")

    def __mod__(self, other):
        return self._execute(other, lambda a, b: a % b, "arithmetic")
    
    def __rmod__(self, other):
        return self._execute(other, lambda a, b: b % a, "arithmetic")

    # binary boolean operations
    def __and__(self, other):
        return self._execute(other, lambda a, b: a and b, "bool_op")

    def __rand__(self, other):
        return self._execute(other, lambda a, b: b and a, "bool_op")

    def __or__(self, other):
        return self._execute(other, lambda a, b: a or b, "bool_op")
    
    def __ror__(self, other):
        return self._execute(other, lambda a, b: b or a, "bool_op")
    
    def __xor__(self, other):
        return self._execute(other, lambda a, b: a ^ b, "bool_op")
    
    def __rxor__(self, other):
        return self._execute(other, lambda a, b: b ^ a, "bool_op")

    # treat as boolean binary operations, using dummy scalar value
    def __invert__(self):
        dummy = True
        return self._execute(dummy, lambda a, _: not a, "bool_op")
    

class DataFrame:
    def __init__(self, data: List[Series], names=List[str]):
        self.data = data
        self.names = names

    def get_column(self, name: str) -> "Series":
        if name not in self.names:
            raise ValueError("Column not found")
        return self.data[self.names.index(name)]
    
    def left_join(self, right_df, column) -> "DataFrame":
        key_to_pos = {k : i for k, i in enumerate(self[column])}
        position_map = right_df._build_position_map(key_to_pos, column)
        new_columns = _scan_right_df(key_to_pos)
        copy = self.data.copy()
        build_new_columns(new_columns, copy)
        return df

    def _build_position_map(self, key_to_pos, column_name):
        column = self[column_name]
        position_map = [None] * len(key_to_pos)
        for i, name in enumerate(column):
            position_map[i] = key_to_pos.get(name, None)
        return position_map



    def __getitem__(self, *args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Series) and arg.dtype == bool:
                serieses = [s[arg] for s in self.data]
                return DataFrame(serieses, self.names)
            elif isinstance(arg, str):
                return self.get_column(arg)
            else:
                raise NotImplemented
        else:
            raise NotImplemented

    def __str__(self):
        transposed = list(zip(*self.data))
        return tabulate(transposed, headers=self.names)

    @classmethod
    def from_dict(cls, data: dict) -> "DataFrame":
        columns = []
        names = []
        n = None
        for name, values in data.items():
            if not (isinstance(values, Iterable) and isinstance(values, Sized)):
                raise ValueError("Values should be array-like")
            if n is None:
                n = len(values)
            if len(values) != n:
                raise ValueError("All columns must have the same length")
            series = Series.from_array_like(values, name=name)
            columns.append(series)
            names.append(name)
        return DataFrame(columns, names)


