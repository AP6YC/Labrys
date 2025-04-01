"""
    test_sets.jl

# Description
The main collection of tests for the `Labrys` project.
This file loads common utilities and aggregates all other unit tests files.
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Labrys
using Logging
using Test

# -----------------------------------------------------------------------------
# DRWATSON MODIFICATIONS TESTS
# -----------------------------------------------------------------------------

@testset "DrWatson Modifications" begin
    # Temp dir for testing
    # test_dir = "testing"
    @assert 1 == 1
    # @info Labrys.work_dir(test_dir)
    # @info Labrys.results_dir(test_dir)
end