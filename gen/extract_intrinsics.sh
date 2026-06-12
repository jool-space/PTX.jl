#!/usr/bin/env bash
# Extract the llvm.nvvm.* intrinsic table from llvm-project at a given tag.
#
# This is the dev-time front half of the registry generator (see DESIGN.md):
# sparse-clone the .td tree, evaluate it with llvm-tblgen, filter to NVVM
# records. The back half (generate_registry.jl, emitting Julia registry
# source) consumes the JSON this produces.
#
# tblgen version note: the tblgen need only understand the tag's TableGen
# *language*, not match its version. LLVM 18's chokes on 22's !listflatten;
# LLVM 21's (from LLVM_full_jll) parses 22.1.7 cleanly and its output count
# agrees with the intrinsic name table embedded in the 22.1.7 llc binary
# (2569 records). Re-verify that agreement on every backend bump — it's the
# cheap conformance check for tblgen-version skew.
#
# Name mapping: most records derive their name as int_nvvm_foo_bar →
# llvm.nvvm.foo.bar (every underscore becomes a dot — even multi-word
# segments like expect_tx → expect.tx), but 353 records carry an explicit
# LLVMName override, typically to *keep* an underscore inside a segment:
# int_nvvm_fence_mbarrier_init_release_cluster →
# llvm.nvvm.fence.mbarrier_init.release.cluster. The generator MUST prefer
# LLVMName when nonempty.
#
# Output schema, per record:
#   record     tblgen record name (int_nvvm_*)
#   llvm_name  explicit LLVMName override, or "" (apply the default rule)
#   ret/params type tokens — a plain string is a tblgen VT name ("i32",
#              "f32", "v2f16", "bf16", "Metadata"); {"ptr": N} is a pointer
#              in address space N; {"match": N} repeats overload slot N;
#              {"any": "pAny"|"iAny"|"fAny"} is an overload slot (the record
#              needs a mangled callsite name — .p3 etc.)
#   props      intrinsic-level properties (IntrConvergent, IntrNoMem, ...)
#   arg_props  per-position properties; "arg" is a 0-based parameter index
#              or "ret" (tblgen's AttrIndex 0 = return is normalized here).
#              Kinds seen at 22.1.7: ImmArg (operand must be an immediate
#              constant in emitted IR), Range {lower, upper(exclusive)},
#              NoCapture, NoAlias, NoUndef, ReadOnly, WriteOnly, and ArgName
#              (operand name from ArgInfo — documentation metadata; ArgInfo's
#              ImmArgPrinter hints are C++-side and deliberately dropped).
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

jq '
  . as $all
  | def rtype:
      $all[.def] as $d
      | if $d.isAny == 1 then {any: $d.VT.def}
        elif ($d["!superclasses"] | index("LLVMQualPointerType")) then {ptr: ($d.Sig[1] // 0)}
        elif ($d["!superclasses"] | index("LLVMMatchType")) then {match: $d.Number}
        else $d.VT.def
        end;
    def rprop:
      if (.def | startswith("anonymous_")) then
        $all[.def] as $d
        | ($d["!superclasses"][1]) as $k
        | (if $d.ArgNo == 0 then "ret" else ($d.ArgNo - 1) end) as $a
        | if $k == "Range" then [{kind: "Range", arg: $a, lower: $d.Lower, upper: $d.Upper}]
          elif $k == "ArgInfo" then
            [ $d.Properties[] | $all[.def]
              | select(."!superclasses" | index("ArgName"))
              | {kind: "ArgName", arg: $a, name: .Name} ]
          else [{kind: $k, arg: $a}]
          end
      else [.def]
      end;
    [ to_entries[]
      | select(.key | startswith("int_nvvm_"))
      | { record: .key,
          llvm_name: .value.LLVMName,
          ret: [.value.RetTypes[] | rtype],
          params: [.value.ParamTypes[] | rtype],
          props: [.value.IntrProperties[] | rprop[] | select(type == "string")],
          arg_props: [.value.IntrProperties[] | rprop[] | select(type == "object")] }
    ]
  | sort_by(.record)' "$WORK/all.json" > "$OUT"

echo "wrote $OUT ($(jq length "$OUT") intrinsics)"
