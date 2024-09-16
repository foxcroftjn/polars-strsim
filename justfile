# run this prior to commit
pre-commit: check format

# run linting checks
check:
    cargo check
    uvx ruff check

# format files
format:
    cargo fmt
    uvx ruff format

# build wheels for linux
build:
    uvx --with 'maturin[patchelf,zig]' maturin build --release --target x86_64-unknown-linux-gnu --zig
    uvx --with 'maturin[patchelf,zig]' maturin build --release --target aarch64-unknown-linux-gnu --zig

# test linux wheels
test-build:
    uv run --no-project --isolated --with polars-strsim --find-links target/wheels python demo.py
