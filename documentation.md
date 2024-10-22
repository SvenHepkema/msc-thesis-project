# ALP on GPU

## Stateless API vs Stateful API

Stateless API, requires rescanning exception vector each time.
```cpp
unalp<uint32_t, float, UnpackingType::LaneArray, UNPACK_N_VECTORS, UNPACK_N_VALUES>
    (out, column, vector_index, lane, value_index);
```

Stateless API, requires user to manage the index, but does not need
to rescan any value in the exception vector.
```cpp
int32_t exception_vector_index = 0;
unalp<uint32_t, float, UnpackingType::LaneArray, UNPACK_N_VECTORS, UNPACK_N_VALUES>
    (out, column, vector_index, lane, value_index, &exception_vector_index);
```

Stateful API. User creates an Unpacker struct to unpack values
from the vector sequentially. User only initializes the struct,
does not need to manage state.

```cpp
auto iterator = 
Unpacker<uint32_t, float, UnpackingType::LaneArray, UNPACK_N_VECTORS, UNPACK_N_VALUES>
    (vector_index, lane, data);

iterator.unpack_next_into(out);
```
