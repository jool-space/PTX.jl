using PTX
using Documenter

DocMeta.setdocmeta!(PTX, :DocTestSetup, :(using PTX); recursive=true)

makedocs(;
    modules=[PTX, PTX.IR, PTX.Codegen, PTX.Parser],
    authors="Anton Oresten <antonoresten@proton.me>",
    sitename="PTX.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/PTX.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Chain DSL" => "dsl.md",
        "Wrappers" => "wrappers.md",
        "Transpiler" => "transpiler.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/PTX.jl",
    devbranch="main",
)
