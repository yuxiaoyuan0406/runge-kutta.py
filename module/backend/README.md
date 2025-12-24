# Module backend

## Usage

### Build

```bash
cd module/backend
bear -- python ./setup.py build_ext --inplace
```

### Import

```python
import numpy as np
import time

import backend

def py_sum_squares(x):
    s = 0.0
    for v in x:
        s += v * v
    return s

x = np.random.rand(5_000_000).astype(np.float64)

backend.Test.sum_squares(x)

t0 = time.perf_counter()
a = py_sum_squares(x)
t1 = time.perf_counter()

t2 = time.perf_counter()
b = backend.Test.sum_squares(x)
t3 = time.perf_counter()

print("py  :", a, "time:", t1 - t0, "s")
print("cpp :", b, "time:", t3 - t2, "s")
print("diff:", abs(a - b))
```
