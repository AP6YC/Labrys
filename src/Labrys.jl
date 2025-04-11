"""
Main module for `Labrys.jl`, a personal Julia package for common machine learning research project tools.

# Imports

The following names are imported by the package as dependencies:
$(IMPORTS)

# Exports

The following names are exported and available when `using` the package:
$(EXPORTS)
"""
module Labrys

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using
    DrWatson,
    DocStringExtensions,
    ProgressMeter,
    NumericalTypeAliases,
    Logging,
    Printf,
    Parameters

greet() = print("Hello World!")


# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

include("lib/lib.jl")

# -----------------------------------------------------------------------------
# DERIVATIVE TYPES AND CONSTANTS
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------

export greet

end # module Labrys
