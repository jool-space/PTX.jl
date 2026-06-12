#!/usr/bin/env bash
# Extract the llvm.nvvm.* intrinsic table from llvm-project at a given tag.
#
# This is the dev-time front half of the registry generator (see DESIGN.md):
# sparse-clone the .td tree, evaluate it with llvm-tblgen, filter to NVVM
# records. The back half (emitting Julia registry source) consumes the JSON
# this produces.
#
# tblgen version note: the tblgen need only understand the tag's TableGen
# *language*, not match its version. LLVM 18's chokes on 22's !listflatten;
# LLVM 21's (from LLVM_full_jll) parses 22.1.7 cleanly and its output count
# agrees with the intrinsic name table embedded in the 22.1.7 llc binary
# (2569 records). Re-verify that agreement on every backend bump — it's the
# cheap conformance check for tblgen-version skew.
#
# Name mapping: most records derive their name as int_nvvm_foo_bar →
# llvm.nvvm.foo.bar, but 353 records (e.g. expect_tx, where an underscore is
# part of a segment) carry an explicit LLVMName override. The generator MUST
# prefer LLVMName when nonempty.
#
# Usage: gen/extract_intrinsics.sh <llvm-tag> [tblgen-path]
#   e.g.: gen/extract_intrinsics.sh llvmorg-22.1.7

set -euo pipefail

TAG="${1:?usage: extract_intrinsics.sh <llvm-tag> [tblgen-path]}"
TBLGEN="${2:-$(julia --project="$(dirname "$0")" -e '
    using LLVM_full_jll
    println(joinpath(LLVM_full_jll.artifact_dir, "tools", "llvm-tblgen"))')}"
WORK="$(mktemp -d)"
OUT="$(dirname "$0")/nvvm_intrinsics_${TAG#llvmorg-}.json"
trap 'rm -rf "$WORK"' EXIT

echo "tag: $TAG"; echo "tblgen: $TBLGEN"
git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none --sparse \
    https://github.com/llvm/llvm-project "$WORK/llvm-project"
git -C "$WORK/llvm-project" sparse-checkout set llvm/include/llvm

"$TBLGEN" --dump-json -I "$WORK/llvm-project/llvm/include" \
    "$WORK/llvm-project/llvm/include/llvm/IR/Intrinsics.td" > "$WORK/all.json"

jq '[ to_entries[]
      | select(.key | startswith("int_nvvm_"))
      | { record: .key,
          llvm_name: .value.LLVMName,
          ret: [.value.RetTypes[].def],
          params: [.value.ParamTypes[].def],
          properties: [.value.IntrProperties[].def] } ]
    | sort_by(.record)' "$WORK/all.json" > "$OUT"

echo "wrote $OUT ($(jq length "$OUT") intrinsics)"
