# Reviewed PTX 9.3 special-register ledger.
#
# PTX §10 defines the registers themselves; §6.4.3 additionally permits the
# .x/.y/.z/.w and .r/.g/.b/.a projections of every .v4 register. Keep the
# versioned source of truth here instead of hand-maintaining one partial list
# in the canonicalizer and another in the transpiler.
#
# The full inventory includes eight bare .v4 roots. They are valid PTX, and
# canonicalization must preserve them, but the current parser/codegen does not
# model vector-valued registers or mov.v4 results. scalar_spellings is the
# deliberately smaller set that codegen can lower safely today.
#
# This is a classification/provenance ledger, not a global parser validator:
# target/version enforcement remains a separate frontend responsibility.
#
# %warpsize is intentionally absent: PTX 9.3 specifies WARP_SZ as an
# immediate constant, not a % special register. The user-facing and codegen
# paths normalize the historical pseudo-register spelling to that immediate.

# PTX 9.3 §4.5.1 says every current target has a 32-thread warp. Keep the
# standard spelling and the legacy pseudo-register spelling separate: only
# WARP_SZ belongs to the PTX source language.
const PREDEFINED_IMMEDIATES = Dict{String, Int}("WARP_SZ" => 32)
const LEGACY_WARP_SIZE_SREG = "%warpsize"

struct SpecialRegisterFamily
    section::String
    introduced::Version
    min_sm::Union{Nothing, Int}  # nothing means every target architecture.
    spellings::Tuple{Vararg{String}}
    scalar_spellings::Tuple{Vararg{String}}
end

function _special_register_family(section::String, introduced::Version,
                                  min_sm::Union{Nothing, Int},
                                  spellings::Tuple{Vararg{String}};
                                  scalar_spellings::Tuple{Vararg{String}} = spellings)
    isempty(spellings) && error("PTX special-register family $section is empty")
    all(startswith(name, "%") for name in spellings) ||
        error("PTX special-register family $section has a non-register spelling")
    length(unique(spellings)) == length(spellings) ||
        error("PTX special-register family $section repeats a spelling")
    all(name -> name in spellings, scalar_spellings) ||
        error("PTX special-register family $section exposes an unknown scalar spelling")
    SpecialRegisterFamily(section, introduced, min_sm, spellings, scalar_spellings)
end

const _THREAD_V4_SREG_ROOTS = (
    "%tid", "%ntid", "%ctaid", "%nctaid",
)
const _CLUSTER_V4_SREG_ROOTS = (
    "%clusterid", "%nclusterid", "%cluster_ctaid", "%cluster_nctaid",
)
const V4_SPECIAL_REG_ROOTS = (
    _THREAD_V4_SREG_ROOTS...,
    _CLUSTER_V4_SREG_ROOTS...,
)
const _V4_COMPONENT_SUFFIXES = (
    ".x", ".y", ".z", ".w", ".r", ".g", ".b", ".a",
)

function _v4_special_register_family(section::String, introduced::Version,
                                     min_sm::Union{Nothing, Int},
                                     roots::Tuple{Vararg{String}})
    spellings = String[]
    scalar_spellings = String[]
    for root in roots
        push!(spellings, root)
        for suffix in _V4_COMPONENT_SUFFIXES
            spelling = root * suffix
            push!(spellings, spelling)
            push!(scalar_spellings, spelling)
        end
    end
    _special_register_family(section, introduced, min_sm, Tuple(spellings);
                             scalar_spellings = Tuple(scalar_spellings))
end

# Each entry is a reviewed PTX 9.3 family. section is intentionally kept
# with the data so additions can be reconciled against a later ISA revision.
const SPECIAL_REGISTER_FAMILIES = (
    # ptx/10-special-registers/{10.1,10.2,10.6,10.7}-special-registers-*.md
    # and ptx/6-instruction-operands/6.4.3-vectors-as-operands.md
    _v4_special_register_family("PTX 9.3 §§10.1, 10.2, 10.6, 10.7; §6.4.3",
                                Version(1, 0), nothing,
                                _THREAD_V4_SREG_ROOTS),
    # ptx/10-special-registers/{10.12,10.13,10.14,10.15}-special-registers-*.md
    # and ptx/6-instruction-operands/6.4.3-vectors-as-operands.md
    _v4_special_register_family("PTX 9.3 §§10.12–10.15; §6.4.3",
                                Version(7, 8), 90,
                                _CLUSTER_V4_SREG_ROOTS),
    # ptx/10-special-registers/10.3-special-registers-laneid.md
    _special_register_family("PTX 9.3 §10.3", Version(1, 3), nothing,
                             ("%laneid",)),
    # ptx/10-special-registers/10.4-special-registers-warpid.md
    _special_register_family("PTX 9.3 §10.4", Version(1, 3), nothing,
                             ("%warpid",)),
    # ptx/10-special-registers/10.5-special-registers-nwarpid.md
    _special_register_family("PTX 9.3 §10.5", Version(2, 0), 20,
                             ("%nwarpid",)),
    # ptx/10-special-registers/10.8-special-registers-smid.md
    _special_register_family("PTX 9.3 §10.8", Version(1, 3), nothing,
                             ("%smid",)),
    # ptx/10-special-registers/10.9-special-registers-nsmid.md
    _special_register_family("PTX 9.3 §10.9", Version(2, 0), 20,
                             ("%nsmid",)),
    # ptx/10-special-registers/10.10-special-registers-gridid.md
    _special_register_family("PTX 9.3 §10.10", Version(1, 0), nothing,
                             ("%gridid",)),
    # ptx/10-special-registers/10.11-special-registers-is_explicit_cluster.md
    _special_register_family("PTX 9.3 §10.11", Version(7, 8), 90,
                             ("%is_explicit_cluster",)),
    # ptx/10-special-registers/10.16-special-registers-cluster_ctarank.md
    _special_register_family("PTX 9.3 §10.16", Version(7, 8), 90,
                             ("%cluster_ctarank",)),
    # ptx/10-special-registers/10.17-special-registers-cluster_nctarank.md
    _special_register_family("PTX 9.3 §10.17", Version(7, 8), 90,
                             ("%cluster_nctarank",)),
    # ptx/10-special-registers/10.18-10.22-special-registers-lanemask-*.md
    _special_register_family("PTX 9.3 §§10.18–10.22", Version(2, 0), 20,
                             ("%lanemask_eq", "%lanemask_le", "%lanemask_lt",
                              "%lanemask_ge", "%lanemask_gt")),
    # ptx/10-special-registers/10.23-special-registers-clock-clock_hi.md
    _special_register_family("PTX 9.3 §10.23 (%clock)", Version(1, 0), nothing,
                             ("%clock",)),
    _special_register_family("PTX 9.3 §10.23 (%clock_hi)", Version(5, 0), 20,
                             ("%clock_hi",)),
    # ptx/10-special-registers/10.24-special-registers-clock64.md
    _special_register_family("PTX 9.3 §10.24", Version(2, 0), 20,
                             ("%clock64",)),
    # ptx/10-special-registers/10.25-special-registers-pm0-pm7.md
    _special_register_family("PTX 9.3 §10.25 (%pm0–%pm3)", Version(1, 3), nothing,
                             Tuple("%pm$i" for i in 0:3)),
    _special_register_family("PTX 9.3 §10.25 (%pm4–%pm7)", Version(3, 0), 20,
                             Tuple("%pm$i" for i in 4:7)),
    # ptx/10-special-registers/10.26-special-registers-pm0_64-pm7_64.md
    _special_register_family("PTX 9.3 §10.26", Version(4, 0), 50,
                             Tuple("%pm$(i)_64" for i in 0:7)),
    # ptx/10-special-registers/10.27-special-registers-envreg32.md
    _special_register_family("PTX 9.3 §10.27", Version(2, 1), nothing,
                             Tuple("%envreg$i" for i in 0:31)),
    # ptx/10-special-registers/10.28-special-registers-globaltimer-globaltimer_lo-globaltimer_hi.md
    _special_register_family("PTX 9.3 §10.28", Version(3, 1), 30,
                             ("%globaltimer", "%globaltimer_lo", "%globaltimer_hi")),
    # ptx/10-special-registers/10.29-special-registers-reserved_smem_offset_begin-reserved_smem_offset_end-reserved_smem_offset_cap-reserved_smem_offset_2.md
    _special_register_family("PTX 9.3 §10.29", Version(7, 6), 80,
                             ("%reserved_smem_offset_begin", "%reserved_smem_offset_end",
                              "%reserved_smem_offset_cap", "%reserved_smem_offset_0",
                              "%reserved_smem_offset_1")),
    # ptx/10-special-registers/10.30-special-registers-total_smem_size.md
    _special_register_family("PTX 9.3 §10.30", Version(4, 1), 20,
                             ("%total_smem_size",)),
    # ptx/10-special-registers/10.31-special-registers-aggr_smem_size.md
    _special_register_family("PTX 9.3 §10.31", Version(8, 1), 90,
                             ("%aggr_smem_size",)),
    # ptx/10-special-registers/10.32-special-registers-dynamic_smem_size.md
    _special_register_family("PTX 9.3 §10.32", Version(4, 1), 20,
                             ("%dynamic_smem_size",)),
    # ptx/10-special-registers/10.33-special-registers-current_graph_exec.md
    _special_register_family("PTX 9.3 §10.33", Version(8, 0), 50,
                             ("%current_graph_exec",)),
    # ptx/10-special-registers/10.34-special-registers-perctamemoryoffset.md
    # Only readable in entries that declare .minperctamemory (§11.4.10).
    _special_register_family("PTX 9.4 §10.34", Version(9, 4), 80,
                             ("%perctamemoryoffset",)),
    # ptx/10-special-registers/10.35-special-registers-perctamemorysize.md
    _special_register_family("PTX 9.4 §10.35", Version(9, 4), 80,
                             ("%perctamemorysize",)),
)

function _special_register_set(field::Symbol)
    names = Set{String}()
    for family in SPECIAL_REGISTER_FAMILIES
        for name in getfield(family, field)
            name in names && error("PTX special-register ledger repeats $name")
            push!(names, name)
        end
    end
    names
end

# PTX 9.4 §10 inventory: 151 spellings = 8 vector roots + 143 scalar or
# projected spellings. The scalar subset is intentionally what codegen uses
# until vector-valued PTX operands have structured lowering support.
const SPECIAL_REGS = _special_register_set(:spellings)
const SCALAR_SPECIAL_REGS = _special_register_set(:scalar_spellings)

# Canonicalization needs the full semantic inventory. Codegen imports the
# scalar subset explicitly so a bare v4 root cannot become malformed scalar
# Julia/inline asm.
