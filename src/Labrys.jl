"""


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


end # module Labrys
