"""
    docstrings.jl

# Description
A collection of common docstrings and docstring templates for the project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# DOCSTRING TEMPLATES
#   These templates tell `DocStringExtensions.jl` how to customize docstrings of various types.
# -----------------------------------------------------------------------------

# Constants template
@template CONSTANTS =
"""
$(FUNCTIONNAME)

# Description
$(DOCSTRING)
"""

# Types template
@template TYPES =
"""
$(TYPEDEF)

# Summary
$(DOCSTRING)

# Fields
$(TYPEDFIELDS)
"""

# Template for functions, macros, and methods (i.e., constructors)
@template (FUNCTIONS, METHODS, MACROS) =
"""
$(TYPEDSIGNATURES)

# Summary
$(DOCSTRING)

# Method List / Definition Locations
$(METHODLIST)
"""

# -----------------------------------------------------------------------------
# DOCSTRING CONSTANTS
#   This location is a collection of variables used for injecting into other docstrings.
# This is useful when many functions utilize the same arguments, etc.
# -----------------------------------------------------------------------------

"""
Docstring prefix denoting that the constant is used as a common docstring element for other docstrings.
"""
const _COMMON_DOC = "Common docstring:"

"""
$_COMMON_DOC the arguments to `DrWatson`-style directory functions.
"""
const _ARG_DRWATSON = """
# Arguments
- `args...`: the string directories to append to the directory.
"""
