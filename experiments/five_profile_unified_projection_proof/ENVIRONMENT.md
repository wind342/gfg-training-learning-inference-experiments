# Environment

The evidence package was reproduced with CPython 3.12 on Windows. Create the
repository's ignored `.venv`, install the repository development dependencies,
then install the frozen W3C parser and PyTorch CPU wheel:

```powershell
python -m venv .venv --system-site-packages
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pip install --require-hashes -r experiments\w3c_prov_projection_v1\requirements.txt
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
```

The PyTorch authority manifest freezes wheel SHA-256
`a8b450c1e58e5800e5b4691dac412f8d2d65a1dc3298166f91596603a3531e6f`.
Install Source Map dependencies from its frozen lock if `node_modules` is not
present:

```powershell
pnpm --dir experiments\source_map_projection install --frozen-lockfile
```

Download and hash-verify the ignored W3C publications and 53 official cases:

```powershell
.venv\Scripts\python.exe -m experiments.w3c_prov_projection_v1.src.bootstrap_references
```

Every exact interpreter, package, lock, profile, crosswalk, and artifact hash
used by the successful run is recorded in `artifacts/unified_manifest.json`.
