#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$script_dir/../static/arch-svg"
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"

for source in "$script_dir"/rivet-{core,data,vision,python}.lini; do
  name="${source##*/}"
  name="${name%.lini}"
  output="$output_dir/$name.svg"
  echo "Generating $output"

  lini --check --strict "$source"
  lini "$source" -o "$output"
done
