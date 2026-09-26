# Preflight development record

The first preflight attempt passed 35/36 unit tests and stopped before creating
the output directory, numerical-case audit, or any formal training result.
The exact formula test detected that NumPy scalar promotion evaluated gamma
in float32: 2.980233091420814e-7 instead of the intended float64
2.9802331269482156e-7 on the dimension-three fixture.

The fix explicitly converts float32 epsilon to a Python float before computing
gamma. The rule, coefficients, thresholds, and test assertions are unchanged.
No test-set outputs or formal training outputs informed this correction.
