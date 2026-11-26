wheel jobs="24" features="":
    @maturin build --release -j {{jobs}} --zig --manylinux 2_28 --features "{{features}}"

# Build the Linux wheel with the same Maturin options as CI.
[script("bash")]
wheel-ci $jobs="24" $features="blas":
    feature_args=()
    if [[ -n "$features" ]]; then
        feature_args=(--features "$features")
    fi

    maturin build --release --locked -j "$jobs" \
        --target x86_64-unknown-linux-gnu --zig --manylinux 2_35 --auditwheel repair \
        --interpreter python3.13 --out dist "${feature_args[@]}"
