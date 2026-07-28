# Ported from pyptx/parser/parser.py (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

const _PTX_94 = Version(9, 4)

const _TARGET_INTRODUCED = let entries = Pair{String,Version}[]
    function add!(version::Version, names...)
        append!(entries, (String(name) => version for name in names))
    end
    add!(Version(1, 0), "10", "11")
    add!(Version(1, 2), "12", "13")
    add!(Version(2, 0), "20")
    add!(Version(3, 0), "30")
    add!(Version(3, 1), "35")
    add!(Version(4, 0), "32", "50")
    add!(Version(4, 1), "37", "52")
    add!(Version(4, 2), "53")
    add!(Version(5, 0), "60", "61", "62")
    add!(Version(6, 0), "70")
    add!(Version(6, 1), "72")
    add!(Version(6, 3), "75")
    add!(Version(7, 0), "80")
    add!(Version(7, 1), "86")
    add!(Version(7, 4), "87")
    add!(Version(7, 8), "89", "90")
    add!(Version(8, 0), "90a")
    add!(Version(8, 6), "100", "100a", "101", "101a")
    add!(Version(8, 7), "120", "120a")
    add!(Version(8, 8), "100f", "101f", "103", "103f", "103a",
                           "120f", "121", "121f", "121a")
    add!(Version(9, 0), "88", "110", "110f", "110a")
    add!(Version(9, 4), "107", "107f", "107a")
    Dict(entries)
end

const _TARGET_OPTIONS = Set(("texmode_unified", "texmode_independent",
                             "debug", "map_f64_to_f32"))
const _TEXTURE_MODES = Set(("texmode_unified", "texmode_independent"))

_version_tuple(v::Version) = (v.major, v.minor)
_version_at_least(v::Version, floor::Version) = _version_tuple(v) >= _version_tuple(floor)
_known_ptx_version(v::Version) = _version_tuple(v) <= _version_tuple(_PTX_94)
_architecture_match(value::String) = match(r"^(sm|compute)_([0-9]+)([af]?)$", value)

function _target_floor(value::String)
    m = _architecture_match(value)
    m === nothing && return nothing
    suffix = String(m.captures[2]) * String(m.captures[3])
    get(_TARGET_INTRODUCED, suffix, nothing)
end

function _validate_target!(s::ParserState, target::Target, version::Version)
    values = target.targets
    isempty(values) && throw(_err(s, ".target requires an architecture"))
    arch = first(values)
    _architecture_match(arch) === nothing &&
        throw(_err(s, "first .target specifier must be an sm_XX or compute_XX architecture, got $(repr(arch))"))

    floor = _target_floor(arch)
    if floor === nothing
        _known_ptx_version(version) &&
            throw(_err(s, "target architecture $(repr(arch)) is not defined by PTX ISA 9.4"))
    elseif !_version_at_least(version, floor)
        throw(_err(s, "target architecture $(repr(arch)) requires PTX ISA $(floor.major).$(floor.minor) or later"))
    end

    options = values[2:end]
    length(unique(options)) == length(options) ||
        throw(_err(s, ".target contains duplicate platform options"))
    for option in options
        if option in _TARGET_OPTIONS
            option == "debug" && !_version_at_least(version, Version(3, 0)) &&
                throw(_err(s, ".target debug requires PTX ISA 3.0 or later"))
            option in _TEXTURE_MODES && !_version_at_least(version, Version(1, 5)) &&
                throw(_err(s, ".target texturing mode requires PTX ISA 1.5 or later"))
        elseif _architecture_match(option) !== nothing
            throw(_err(s, ".target specifies more than one architecture"))
        elseif _known_ptx_version(version)
            throw(_err(s, "unknown PTX 9.4 .target option $(repr(option))"))
        end
    end
    count(in(_TEXTURE_MODES), options) <= 1 ||
        throw(_err(s, ".target cannot select both texturing modes"))

    # PTX 11.1.2 explicitly disallows this legacy mapping on sm_13.
    base_arch = replace(arch, "compute_" => "sm_")
    base_arch == "sm_13" && "map_f64_to_f32" in options &&
        throw(_err(s, "map_f64_to_f32 is not allowed for sm_13"))
    target
end

function _validate_module_target_options!(s::ParserState, initial::Target,
                                          directives::Tuple{Vararg{Statement}})
    targets = Target[initial]
    append!(targets, (d.target for d in directives if d isa TargetDirective))
    texture_modes = String[]
    for target in targets, option in target.targets[2:end]
        option in _TEXTURE_MODES && push!(texture_modes, option)
    end
    length(unique(texture_modes)) <= 1 ||
        throw(_err(s, "the module-wide .target texturing mode cannot be changed"))
    nothing
end
