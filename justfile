wheel jobs="24" features="":
    @maturin build --release -j {{jobs}} --zig --manylinux 2_28 --features "{{features}}"
