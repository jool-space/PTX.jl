using PTX.Parser: parse as parse_ptx, ParseError
using PTX.IR: Module, Function, Instruction, RawLine, format

# Ablation tests: each fixture in `test/corpus/corrupted/` is a single-mutation
# variant of a clean corpus file. The test asserts the expected outcome per
# mutation, locking in the parser's contract:
#
#   - `:parse_error` — header-level malformations throw ParseError. No
#                      RawLine fallback above the header line.
#   - `:rawline_top` — body-level structural failures (post-header) parse;
#                      the broken line lands in a top-level RawLine and the
#                      round-trip is byte-identical.
#   - `:rawline_body` — same, but the RawLine is inside a Function body
#                       (matched braces; only the inner instruction broke).
#   - `:permissive`  — the input "looks wrong" but is structurally a valid
#                      Instruction (e.g. unknown opcode). Parser does NOT
#                      whitelist opcodes; ptxas is the validity gate.
#
# Source files are shipped verbatim in `test/corpus/corrupted/`; the manifest
# below is the source of truth for expected outcomes. Each `assert` callback
# can add fixture-specific structural checks.

const _CORRUPTED_DIR = joinpath(@__DIR__, "..", "corpus", "corrupted")

# Walk a Module's IR for RawLine fallback nodes at every nesting depth. Keep
# the top/body split for this fixture manifest, but do not mistake a direct
# Function.body scan for a complete inventory: scoped Blocks and
# IntrinsicScopes can contain fallback nodes too.
function _count_rawlines(m::Module)
    paths = _rawline_paths(m)
    top = count(entry -> !occursin(".body[", first(entry)), paths)
    body = length(paths) - top
    (top = top, body = body)
end

const _ABLATION_MANIFEST = [
    (file = "minimal__drop_version.ptx",
     outcome = :parse_error,
     assert  = nothing),

    (file = "minimal__bad_version_minor.ptx",
     outcome = :parse_error,
     assert  = nothing),

    (file = "minimal__bad_address_size.ptx",
     outcome = :parse_error,
     assert  = nothing),

    (file = "vector_add__drop_open_brace.ptx",
     outcome = :rawline_top,
     # Parser recovers aggressively: it still matches `.entry vector_add(...)`
     # as a Function (the param list closes cleanly), gives it an empty body,
     # then re-enters top-level parsing — so all the instructions end up as
     # top-level directives and the stray closing `}` lands in RawLine.
     assert  = m -> let funs = filter(d -> d isa Function, m.directives)
                       length(funs) == 1 && isempty(funs[1].body) &&
                       count(d -> d isa Instruction, m.directives) >= 15
                    end),

    (file = "vector_add__missing_semi.ptx",
     outcome = :rawline_body,
     # Function parses; only the missing-semi line lands in RawLine inside
     # the body. The other 19 instructions parse normally.
     assert  = m -> let f = first(filter(d -> d isa Function, m.directives))
                       count(s -> s isa Instruction, f.body) >= 18 &&
                       count(s -> s isa RawLine, f.body) == 1
                    end),

    (file = "vector_add__garbage_opcode.ptx",
     outcome = :permissive,
     # The bogus opcode `xyzzy.f32` parses as a normal Instruction with
     # opcode = "xyzzy". Number of Instructions matches the original
     # vector_add (20 in the body); no RawLine.
     assert  = m -> let f = first(filter(d -> d isa Function, m.directives))
                       any(s -> s isa Instruction && s.opcode == "xyzzy", f.body) &&
                       count(s -> s isa RawLine, f.body) == 0
                    end),
]

@testset "ablation: $(case.file)" for case in _ABLATION_MANIFEST
    path = joinpath(_CORRUPTED_DIR, case.file)
    src  = read(path, String)

    if case.outcome === :parse_error
        @test_throws ParseError parse_ptx(src)
    else
        m = parse_ptx(src)
        rl = _count_rawlines(m)
        if case.outcome === :rawline_top
            @test rl.top >= 1
        elseif case.outcome === :rawline_body
            @test rl.body >= 1
            @test rl.top == 0
        elseif case.outcome === :permissive
            @test rl.top == 0
            @test rl.body == 0
        else
            error("unknown outcome :$(case.outcome)")
        end
        # Round-trip preserves the (corrupted) bytes exactly — the whole
        # point of the RawLine escape and the raw_source short-circuit.
        @test format(m) == src
        case.assert === nothing || @test case.assert(m)
    end
end
