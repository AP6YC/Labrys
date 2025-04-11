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




# DEV CONSTANTS

"""
$_COMMON_DOC config filename argument.
"""
const ARG_CONFIG_FILE = """
- `config_file::AbstractString`: the config file name as a string.
"""

"""
$_COMMON_DOC config dictionary argument.
"""
const ARG_CONFIG_DICT = """
- `config::ConfigDict`: the config parameters as a dictionary.
"""

"""
$_COMMON_DOC argument for a split ratio `p`.
"""
const ARG_P = """
- `p::Float`: kwarg, the split ratio ∈ `(0, 1)`, default $(DEFAULT_P).
"""

"""
$_COMMON_DOC argument for a training dataset shuffle flag.
"""
const ARG_SHUFFLE = """
- `shuffle::Bool`: flag for shuffling the data, default $(DEFAULT_SHUFFLE).
"""

"""
$_COMMON_DOC argument for an existing `Plots.Plot` object to plot atop.
"""
const ARG_PLOT = """
- `p::Plots.Plot`: an existing `Plots.Plot` object.
"""

"""
$_COMMON_DOC argument for a file name.
"""
const ARG_FILENAME = """
- `filename::AbstractString`: the full file path as a string.
"""

"""
$_COMMON_DOC argument for a directory function
"""
const ARG_SIM_DIR_FUNC = """
- `dir_func::Function`: the function that provides the correct file path with provided strings.
"""

"""
$_COMMON_DOC argument for the simulation options dictionary.
"""
const ARG_SIM_D = """
- `d::AbstractDict`: the simulation options dictionary.
"""

"""
$_COMMON_DOC argument for additional simulation options.
"""
const ARG_SIM_OPTS = """
- `opts::AbstractDict`: additional options for the simulation.
"""

"""
$_COMMON_DOC argument for [`DataSplit`](@ref).
"""
const ARG_DATASPLIT = """
- `data::DataSplit`: a [`DataSplit`](@ref) container of a supervised train/test split.
"""

"""
$_COMMON_DOC argument for [`SupervisedDataset`](@ref).
"""
const ARG_SUPERVISEDDATASET = """
- `data::SupervisedDataset`: a [`SupervisedDataset`](@ref) containing samples and their labels.
"""

"""
$_COMMON_DOC argument for the number of classes.
"""
const ARG_N_CLASS = """
- `n_class::Int=0`: the true number of classes (if known).
"""

"""
$_COMMON_DOC argument for an index parameter.
"""
const ARG_INDEX = """
- `index::Int`: the element index.
"""

"""
$_COMMON_DOC argument for input data of arbitrary dimension.
"""
const ARG_X = """
- `x::RealArray`: the input data.
"""

"""
$_COMMON_DOC argument for a [`MultiHeadField`](@ref).
"""
const ARG_MULTIHEADFIELD = """
- `field::MultiHeadField`: the [`MultiHeadField`](@ref) object.
"""

"""
$_COMMON_DOC argument for a [`DeepHeadART`](@ref).
"""
const ARG_DEEPHEADART = """
- `art::DeepHeadART`: the [`DeepHeadART`](@ref) module.
"""

"""
$_COMMON_DOC argument for an `ARTModule`.
"""
const ART_ARG_DOCSTRING = """
- `art::ARTModule`: the ARTModule module.
"""

"""
$_COMMON_DOC argument for a sample 'x'.
"""
const X_ARG_DOCSTRING = """
- `x::RealVector`: the input sample vector to use.
"""

"""
$_COMMON_DOC argument for a label 'y'.
"""
const ARG_Y = """
- `y::Integer`: the label for the input sample.
"""

"""
$_COMMON_DOC argument for a weight vector 'W'.
"""
const W_ARG_DOCSTING = """
- `W::RealVector`: the weight vector to use.
"""

"""
$_COMMON_DOC shared arguments string for methods using an ART module, sample 'x', and weight vector 'W'.
"""
const ART_X_W_ARGS = """
# Arguments
$(ART_ARG_DOCSTRING)
$(X_ARG_DOCSTRING)
$(W_ARG_DOCSTING)
"""

"""
$_COMMON_DOC argument for a [`CommonARTModule`](@ref).
"""
const ARG_COMMONARTMODULE = """
- `art::CommonARTModule`: the [`CommonARTModule`](@ref) model.
"""

"""
$_COMMON_DOC argument for a [`DeepARTModule`](@ref) for training or testing.
"""
const ARG_DEEPARTMODULE = """
- `art::DeepARTModule`: the [`DeepARTModule`](@ref) model.
"""

"""
$_COMMON_DOC argument for task-incremental data splits implemented as a [`ClassIncrementalDataSplit`](@ref).
"""
const ARG_TIDATA = """
- `tidata::ClassIncrementalDataSplit`: the task-incremental data split.
"""

"""
$_COMMON_DOC argument for the number of training samples to use.
"""
const ARG_N_TRAIN = """
- `n_train::Integer`: the number of training iterations.
"""

"""
$_COMMON_DOC argument for the number of testing samples to use.
"""
const ARG_N_TEST = """
- `n_test::Integer`: the number of testing iterations.
"""
