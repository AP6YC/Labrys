using Labrys
using Documenter

DocMeta.setdocmeta!(Labrys, :DocTestSetup, :(using Labrys); recursive=true)

makedocs(;
    modules=[Labrys],
    authors="Sasha Petrenko <sap625@mst.edu> and contributors",
    sitename="Labrys.jl",
    format=Documenter.HTML(;
        canonical="https://AP6YC.github.io/Labrys.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/AP6YC/Labrys.jl",
    devbranch="main",
)
