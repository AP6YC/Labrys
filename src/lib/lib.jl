"""
    lib.jl

# Description
This file aggregates the library code from other files for the `DeepART` project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Constants definitions
include("constants.jl")

# Docstring variables
include("docstrings.jl")

# DrWatson wrapper functions
include("drwatson.jl")

# Data module
include("data/lib.jl")
