# This file is part of KWIVER, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

"""cpp_to_pybind11.py:

Reads a C++ header file and creates C++ files with pybind11 bindings for
the objects defined in the input file. cpp_to_pybind11.py uses pygccxml to parse
the input C++, and the generated C++ file contains functions used for generating
bindings.

To install the required dependencies:

python -v venv env
source env/bin/activate
pip install pygccxml castxml

This script was derived from https://gitlab.kitware.com/cmb/smtk/-/blob/master/utilities/python/cpp_to_pybind11.py
"""

import argparse
import configparser
import os
import shutil
import sys
from pathlib import Path
import logging
import time
import subprocess

logger = logging.getLogger(__name__)


# Find out the file location within the sources tree
this_module_dir_path = os.path.abspath(os.path.dirname(sys.modules[__name__].__file__))
# Add pygccxml package to Python path
sys.path.append(os.path.join(this_module_dir_path, "..", ".."))

from pygccxml import parser  # nopep8
from pygccxml import declarations  # nopep8
from pygccxml import utils  # nopep8

LICENSE_STRING = """// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.
"""

COMMAND_STRING = """// Generated using: """


def tranform_default_value(arg):
    """
    A handcrafted list of values that we should not use as default in python.
    This includes values that are not automatically convertable in python.
    """
    if arg.default_value is None:
        return None
    elif declarations.type_traits.is_fundamental(
        declarations.type_traits.remove_elaborated(arg.decl_type)
    ):
        # fundamental values are handled automatically by pybind
        return arg.default_value
    elif arg.default_value == "std::cout":
        # std::cout cannot be simply converted to a python object
        return None
    elif declarations.smart_pointer_traits.is_smart_pointer(arg.decl_type):
        # we accept only nullptrs smart_pointers which we convert to None
        # Add more types here to have them converted to None
        if arg.default_value == "kwiver::vital::image_container_sptr()":
            return "py::none()"
    elif declarations.container_traits.vector_traits.is_my_case(arg.decl_type):
        # accept only vector of fundemental types
        value_type = declarations.container_traits.vector_traits.element_type(
            arg.decl_type
        )
        if declarations.type_traits.is_fundamental(value_type):
            return arg.default_value
        return None
    return None


def get_scope(namespace):
    """
    Returns a list of namespaces from global to the input namespace
    """
    scope = []
    current_namespace = namespace.parent
    while current_namespace != namespace.top_parent:
        scope.append(current_namespace.name)
        current_namespace = current_namespace.parent

    scope = scope[::-1]
    return scope


def full_class_name(class_):
    """
    Returns the absolute name of a class, with all nesting namespaces included
    """
    return "::".join(get_scope(class_) + [class_.name])


def mangled_name(obj):
    """
    Returns a string that is unique to the object and can be used as a variable
    name
    """
    return (
        "_".join(get_scope(obj) + [obj.name])
        .replace("<", "_")
        .replace(">", "_")
        .replace("::", "_")
    )


def get_path_relative_to_directory(path, directory):
    return os.path.relpath(
        os.path.abspath(path),
        os.path.commonprefix([os.path.abspath(directory), os.path.abspath(path)]),
    )


def create_trampoline(filename, project_source_directory, algo_namespace, stream):
    fileguard = Path(filename).stem.replace(".", "_").upper() + "_TRAMPOLINE_TXX"

    # output file preamble
    stream(LICENSE_STRING)
    stream(COMMAND_STRING)
    stream(f"#ifndef {fileguard}")
    stream(f"#define {fileguard}")
    stream("")

    # includes
    stream("#define KWIVER_PYBIND11_INCLUDE")
    stream("#include <pybind11/pybind11.h>")
    stream("#include <python/kwiver/vital/algo/algorithm_trampoline.txx>")
    stream(
        "#include <%s>"
        % os.path.relpath(
            os.path.abspath(filename),
            os.path.commonprefix(
                [os.path.abspath(project_source_directory), os.path.abspath(filename)]
            ),
        )
    )
    stream("")
    class_name = Path(filename).stem
    try:
        (class_,) = algo_namespace.classes(class_name)
    except RuntimeError as error:
        logger.error(f"Could not find {class_name} is kwiver::vital::algo namespace!")
        logger.error(
            "Make sure the proper header and declarations are given to the parser"
        )
        logger.error(error)
        exit(1)

    # namespace
    stream("namespace kwiver::vital::python {")

    # generate trampoline class
    assert class_ is not None
    stream(
        f"""
template< class {class_.name}_base = {full_class_name(class_)} >
class {class_.name}_trampoline
    : public algorithm_trampoline< {class_.name}_base >
{{
  public:
    using algorithm_trampoline< {class_.name}_base >::algorithm_trampoline;
"""
    )

    stream('#pragma GCC diagnostic ignored "-Wdeprecated-declarations"')

    ## generate only the methods required for a trampoline class ie pure virtual
    for member in [
        *class_.private_members,
        *class_.protected_members,
        *class_.public_members,
    ]:
        if member.__class__.__name__ != "member_function_t":
            continue
        if member.virtuality != declarations.VIRTUALITY_TYPES.NOT_VIRTUAL:
            pybind11_macro = "PYBIND11_OVERLOAD"
            if member.virtuality == declarations.VIRTUALITY_TYPES.PURE_VIRTUAL:
                pybind11_macro = "PYBIND11_OVERLOAD_PURE"
            elif member.access_type == "private":
                pybind11_macro = "PYBIND11_OVERLOAD_PURE"  # This looks like a kwiver pattern: private non pure virtual methods become accessible by making overlading them as pure
            argument_string = ", ".join(
                [arg.decl_type.decl_string + " " + arg.name for arg in member.arguments]
            )
            argument_name_list = ", ".join([arg.name for arg in member.arguments])
            const_specifier = "const " if member.has_const else ""
            return_type_str = str(member.return_type)
            # If the return type contains a comma (e.g., std::map<K, V>),
            # use a type alias to avoid confusing the preprocessor macro
            if "," in return_type_str:
                macro_return_type = f"{member.name}_return_t"
                alias_line = f"\n    using {macro_return_type} = {return_type_str};"
            else:
                macro_return_type = return_type_str
                alias_line = ""
            stream(
                f"""
  {return_type_str}
  {member.name}({argument_string}) {const_specifier}override
  {{{alias_line}
    {pybind11_macro}(
      {macro_return_type},
      {full_class_name(class_)},
      {member.name},
      {argument_name_list}
      );
  }}"""
            )

    stream("}; // class")
    stream("} // namespace")
    stream("#undef KWIVER_PYBIND11_INCLUDE")
    stream("#endif")

    return {class_: f"{class_.name}_trampoline"}


def create_class_header(filename, project_source_directory, stream):
    """
    Generate a header for the pybind11 bindings of the class described in filename
    """
    header_path_relative_to_project = get_path_relative_to_directory(
        filename, project_source_directory
    )
    fileguard = (
        "KWIVER_PYTHON_"
        + header_path_relative_to_project.replace(".", "_")
        .replace(os.sep, "_")  # replace directory separator from filenames
        .replace("/", "_")  # on windows there is more than one separator
        .replace("-", "_")
        .upper()
    )
    class_name = Path(filename).stem

    stream(LICENSE_STRING)
    stream(COMMAND_STRING)
    stream(f"#ifndef {fileguard}")
    stream(f"#define {fileguard}")
    stream("")
    stream("#include <pybind11/pybind11.h>")
    stream("")
    stream("namespace kwiver::vital::python {")
    stream("namespace py = pybind11;")
    stream("")

    stream(f"void {class_name}(py::module& m);")
    stream("}")
    stream("#endif")


def create_class_implementation(
    filename,
    project_source_directory,
    algo_namespace,
    stream,
):
    # output file preamble
    header_path_relative_to_project = os.path.relpath(
        os.path.abspath(filename),
        os.path.commonprefix(
            [os.path.abspath(project_source_directory), os.path.abspath(filename)]
        ),
    )

    # The trampoline sits beside the generated sources, which is
    # python/kwiver/vital/algo whatever the interface header's own path is.
    # Those two used to be the same thing; VIAME's phase 5 moved the headers
    # to library/ and they are not any more.
    trampoline_path = (
        Path("python/kwiver/vital/algo")
        / Path(Path(filename).stem + "_trampoline.txx")
    )

    stream(LICENSE_STRING)
    stream(COMMAND_STRING)
    stream("")
    stream("#define KWIVER_PYBIND11_INCLUDE")
    stream("#include <viame/core_types/casters.h>")
    stream("#include <pybind11/pybind11.h>")

    stream("")
    stream(f"#include <{header_path_relative_to_project}>")
    stream(f"#include <python/kwiver/vital/algo/algorithm.txx>")
    stream(f"#include <{trampoline_path}>")
    stream("")
    # namespace
    stream("namespace kwiver::vital::python {")
    stream("namespace py = pybind11;")
    stream("")

    cpp_class_name = Path(filename).stem
    stream(f"void {cpp_class_name}(py::module& m)")
    stream("{")
    stream('  py::module::import("kwiver.vital.config");')
    stream('  py::module::import("kwiver.vital.types");')

    (class_,) = algo_namespace.classes(cpp_class_name)
    parse_class(class_, stream)

    stream("")

    stream("}")
    stream("#undef KWIVER_PYBIND11_INCLUDE")


def copy_handwritten_trampoline(filename, directory, destination):
    """Use a hand-written trampoline instead of generating one.

    `PYBIND11_OVERLOAD` passes every argument to python by value. That is
    right for an input, and wrong for the in/out parameters three of these
    interfaces have -- `extract_descriptors` may replace the feature set it
    is given, and both estimators fill an inlier flag per point. A python
    implementation of any of the three can set those all it likes and the
    C++ caller sees nothing, which is silent and total: a `match_features`
    filtered by inliers keeps none of its matches.

    Expressing that in the generator would mean teaching it a convention for
    what a python method returns when the C++ signature has an out
    parameter, for the three cases in the tree that have one. A file per
    interface, written once, says it more plainly. Phase 8 replaces the
    generator wholesale (P8-T02); until then this is where the difference
    lives.

    Returns True if one was copied.
    """
    if not directory:
        return False

    source = Path(directory) / (Path(filename).stem + "_trampoline.txx")

    if not source.exists():
        return False

    logger.debug(f"hand-written trampoline: {source}")
    shutil.copyfile(source, destination)
    return True


def parse_file(
    algo_namespace,
    filename,
    project_source_directory,
    output_basename,
    handwritten_trampoline_directory="",
):
    """
    Entry point for parsing a file
    """

    logger.debug(f"{filename=}")

    class_header = output_basename + ".h"
    class_impl = output_basename + ".cxx"
    class_trampoline = output_basename + "_trampoline.txx"

    handwritten = copy_handwritten_trampoline(
        filename, handwritten_trampoline_directory, class_trampoline
    )

    with open(class_header, "w") as h, open(class_impl, "w") as cxx:
        stream_header = stream_with_line_breaks(h)
        stream_impl = stream_with_line_breaks(cxx)

        if not handwritten:
            with open(class_trampoline, "w") as tramp:
                create_trampoline(
                    filename, project_source_directory, algo_namespace,
                    stream_with_line_breaks(tramp)
                )

        create_class_header(filename, project_source_directory, stream_header)
        create_class_implementation(
            filename,
            project_source_directory,
            algo_namespace,
            stream_impl,
        )


def derive_python_name(class_):
    # start with camelCase and handle special cases
    python_name = "".join(part.title() for part in class_.name.split("_"))
    # special case class_io  -> ClassIo -> ClassIO
    if python_name[-2:] == "Io":
        python_name = python_name[:-1] + "O"
    # special case estimate_pnp  -> EstimatePnp -> EstimatePNP
    elif class_.name == "estimate_pnp":
        python_name = "EstimatePNP"
    # special case uuid_factory  -> UuidFactory -> UUIDFactory
    elif class_.name == "uuid_factory":
        python_name = "UUIDFactory"
    elif class_.name == "uv_unwrap_mesh":
        python_name = "UVUnwrapMesh"
    return python_name


def parse_class(class_, stream):
    """
    Write bindings for a class
    """

    full_class_name_ = full_class_name(class_)
    python_class_name = derive_python_name(class_)
    stream(
        f"""
    py::class_<{full_class_name_},
               std::shared_ptr<{full_class_name_}>,
               kwiver::vital::algorithm,
               {class_.name}_trampoline<> > instance(m,  "{python_class_name}");
    """
    )

    stream("    instance")

    if class_.is_abstract:
        stream("    .def(py::init<>())")
    else:
        for constructor in class_.constructors():
            if constructor.parent.name != class_.name:
                continue
            if constructor.access_type != "public":
                continue
            # create a string of all constructor arguments along with their default value
            types = ", ".join([str(arg.decl_type) for arg in constructor.arguments])
            names = ", ".join(
                [
                    f'py::arg("{arg.name}") = {arg.default_value}'
                    for arg in constructor.arguments
                ]
            )
            stream(f"    .def(py::init<{types}>(), {names})")

    all_methods = set()
    overloaded_methods = set()
    virtual_methods = set()
    for member in class_.public_members:
        if member.parent.name != class_.name:
            continue
        if member.__class__.__name__ != "member_function_t":
            continue
        if member.virtuality == declarations.VIRTUALITY_TYPES.VIRTUAL:
            virtual_methods.add(member.name)
        if member.name in all_methods:
            overloaded_methods.add(member.name)
        else:
            all_methods.add(member.name)

    # this code block separates method pairs that look like "get_XXX()" and
    # "set_xxxx()" and flags them to be used as set-s and get-s to define a
    # property.
    use_properties = True
    property_set_methods = set()
    property_get_methods = set()
    if use_properties:
        property_set_methods = set(
            [
                m
                for m in class_.public_members
                if m.name[:4] == "set_" and m.name not in virtual_methods
            ]
        )
        property_get_methods = set(
            [
                m
                for m in class_.public_members
                if m.name[:4] == "get_" and m.name not in virtual_methods
            ]
        )

        for getter, setter in zip(property_get_methods, property_set_methods):
            name = getter.name[4:]
            stream(
                f'    .def_property("{name}", &{full_class_name_}::{getter.name}, &{full_class_name_}::{setter.name})'
            )

    for member in class_.public_members:
        if member.parent.name != class_.name:
            continue
        if member.__class__.__name__ != "member_function_t":
            # we are interested only in methods for now
            continue
        if member in property_get_methods and member in property_set_methods:
            # already registered
            continue
        static = "_static" if member.has_static else ""
        const = " const" if member.has_const else ""
        doc = (
            (
                ", "
                + 'py::doc(R"('
                + "\n".join(line for line in member.comment.text).replace("///", "")
                + ')")'
            )
            if len(member.comment.text) != 0
            else ""
        )
        if len(member.arguments) == 0:
            args_ = ""
        else:
            args_list = []
            for arg in member.arguments:
                default_value = tranform_default_value(arg)
                arg_str = (
                    f'py::arg("{arg.name}") = {tranform_default_value(arg)}'
                    if default_value is not None
                    else f'py::arg("{arg.name}")'
                )
                args_list.append(arg_str)
            args_ = ", " + ", ".join(args_list)
        if member.name in overloaded_methods:
            args_declaration = ", ".join(
                [arg.decl_type.decl_string for arg in member.arguments]
            )
            method_specifier = "" if member.has_static else full_class_name_ + "::"
            stream(
                f'    .def{static}("{member.name}", ({member.return_type} ({method_specifier}*)({args_declaration}){const}) &{full_class_name_}::{member.name}{doc}{args_})'
            )
        else:
            stream(
                f'    .def{static}("{member.name}", &{full_class_name_}::{member.name}{doc}{args_})'
            )

    for variable in class_.variables(allow_empty=True):
        if variable.parent.name != class_.name:
            continue

        if variable.access_type == "public":
            static = "_static" if variable.type_qualifiers.has_static else ""
            if declarations.type_traits.is_const(variable.decl_type):
                stream(
                    f'    .def_readonly{static}("{variable.name}", &{full_class_name_}::{variable.name})'
                )

    stream("    ;")
    stream(f"  register_algorithm< {full_class_name_} > (instance);")
    stream("}")


def parse_headers(
    headers,
    include_directories,
    declaration_names,
    run_external_castxml=None,
    compiler_path=None,
    compiler_include_dirs=(),
):
    """
    Parse the provided headers and extract declatation form them.
    `include_directories` should provide all the paths required to resolved
    `includes` in the headers.  If declaration_names is not empty. Only the
    classes in `declaration_names` will be considered which speeds up the
    process significantly.
    """
    # Find out the xml generator (gccxml or castxml)
    generator_path, generator_name = utils.find_xml_generator()
    cflags = "-std=c++17 -fsized-deallocation -DKWIVER_PYBIND11_WRAPPING"

    ## Configure the xml generator
    config = parser.xml_generator_configuration_t(
        start_with_declarations=declaration_names,
        include_paths=include_directories,
        xml_generator_path=generator_path,
        xml_generator=generator_name,
        cflags=cflags,
        compiler_path=compiler_path,
        castxml_epic_version=1,  # required to be able to parse comments
    )

    toc = time.perf_counter()
    if run_external_castxml:
        # compose a file will all the headers this has the same effect with COMPILATION_MODE.ALL_AT_ONCE
        file = ""
        for header in headers:
            file += f'#include "{header}" \n'
        with open("python_kwiver_vital_algo.h", "w") as F:
            F.write(file)

        # compose the command for castxml
        command = []
        command.append(generator_path)
        command.append(cflags)
        command.append(" ".join(f'-I"{path}"' for path in include_directories))
        # castxml parses with its own bundled clang, which looks for a GCC
        # installation in the usual prefixes. That misses a standard library
        # kept anywhere else -- a Red Hat gcc-toolset under /opt/rh, say, where
        # nothing but <cstdlib> not found comes back. Point it at the include
        # directories the compiler doing the build actually uses.
        command.append(
            " ".join(f'-isystem "{path}"' for path in compiler_include_dirs)
        )
        command.append("-c -x c++")
        if compiler_path:
            command.append(f'--castxml-cc-msvc  "(" "{compiler_path}"  -std:c++17  ")"')
        # With no compiler_path we ask castxml to emulate nothing and parse
        # with its own bundled clang, which is the only combination that reads
        # a modern glibc without tripping over the _FloatN types.
        command.append(f"--castxml-output=1")
        command.append(f"-o python_kwiver_vital_algo.xml python_kwiver_vital_algo.h")
        command.append('--castxml-start "{0}"'.format(",".join(declaration_names)))
        logger.debug(command)

        cmd_line = " ".join(command)
        print(cmd_line)
        process = subprocess.Popen(args=cmd_line, shell=True, stdout=subprocess.PIPE)

        process.wait()
        if process.returncode != 0:
            # Without this the missing xml surfaces later as a bare "xml file
            # does not exist" from pygccxml, with castxml's actual diagnostics
            # nowhere in sight.
            raise RuntimeError(
                f"CASTXML failed with exit code {process.returncode}: {cmd_line}"
            )
        parsed_declarations = parser.parse_xml_file(
            "python_kwiver_vital_algo.xml", config=config
        )
    else:
        parsed_declarations = parser.parse(
            [str(header) for header in headers],
            config,
            compilation_mode=parser.COMPILATION_MODE.ALL_AT_ONCE,
        )

    tic = time.perf_counter()
    logger.debug(f"Time to parse {toc-tic}")
    return parsed_declarations


def parse_arguments():
    global COMMAND_STRING
    config_parser = argparse.ArgumentParser(
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    config_parser.add_argument(
        "-c",
        "--config",
        help="Use config file",
        metavar="FILE",
    )
    args, _ = config_parser.parse_known_args()

    defaults = None
    if args.config:
        config = configparser.ConfigParser()
        config.read(args.config)
        defaults = dict(config.items("Defaults"))

    arg_parser = argparse.ArgumentParser(parents=[config_parser])
    arg_parser.add_argument(
        "-I",
        "--include-dirs",
        help="Add an include directory to the parser",
        default="",
    )

    arg_parser.add_argument(
        "--compiler-include-dirs",
        help="The build compiler's own implicit include directories, passed to "
        "castxml as -isystem so it finds that compiler's standard library",
        default="",
    )

    arg_parser.add_argument(
        "-d",
        "--declaration-names",
        help="names of C++ classes to extract",
        default="",
    )

    input_arg = arg_parser.add_argument(
        "-i", "--input", help="<Required> Input C++ header file", required=True
    )

    output_arg = arg_parser.add_argument(
        "-o", "--output", help="Basename of output C++ file(s)", required=True
    )

    arg_parser.add_argument(
        "-s", "--project-source-directory", help="Project source directory", default="."
    )
    arg_parser.add_argument(
        "-w",
        "--working-directory",
        help="Working directory. All generated files will be written here.",
        default=".",
    )

    arg_parser.add_argument(
        "--classes",
        help="If specificed, input, output and declaration_name args will be derived by these names combining the values of `base-directory-for-input` and `working-directory`",
        default=None,
    )
    arg_parser.add_argument(
        "--common-namespace",
        help="If specificed, the namespace will be prepended in all classes given in class-basename argument",
        default=None,
    )
    arg_parser.add_argument(
        "--base-directory-for-input",
        help="Base directory for all files in input files. If not given, input is expected to be relative paths with respect to project source argument",
        default=None,
    )
    arg_parser.add_argument(
        "--list-delimiter",
        help="Character used to separate lists in .ini file /commandline",
        default="\n",
    )
    arg_parser.add_argument(
        "--run-external-castxml",
        help="Compose the command and execute castxml as part of this script (only supported on windows)",
        default=False,
    )
    arg_parser.add_argument(
        "--compiler-path",
        help="Path to native compiler, required if run-external-castxml is selected",
    )
    arg_parser.add_argument(
        "--handwritten-trampoline-directory",
        help="Directory of hand-written <class>_trampoline.txx files. One "
             "found there is copied in place of the generated trampoline, "
             "for the interfaces whose signature the generator cannot "
             "express -- an in/out parameter, say.",
        default="",
    )

    arg_parser.add_argument(
        "-v", "--verbose", help="Print out generated wrapping code", action="store_true"
    )
    if defaults:
        arg_parser.set_defaults(**defaults)
        if "classes" in defaults.keys():
            output_arg.required = False
            input_arg.required = False
        if "output" in defaults.keys():
            output_arg.required = False
        if "input" in defaults.keys():
            input_arg.required = False

    logger.debug(defaults)
    args = arg_parser.parse_args()

    command_args = " ".join(sys.argv)
    COMMAND_STRING += command_args
    logger.debug(args)
    sep = args.list_delimiter

    if args.classes:
        args.classes = args.classes.split(sep)
        if args.base_directory_for_input:
            args.input = [
                Path(args.base_directory_for_input) / str(name.split("::")[-1] + ".h")
                for name in args.classes
            ]
        else:
            args.input = [name.split("::")[-1] + ".h" for name in args.classes]
        args.output = args.classes
        if args.common_namespace:
            args.declaration_names = [
                args.common_namespace + "::" + name for name in args.classes
            ]
        else:
            args.declaration_names = args.classes
    else:
        args.input = args.input.split(sep)
        args.output = args.output.split(sep)
        args.declaration_names = args.declaration_names.split(sep)

    wd = Path(args.working_directory).absolute()
    sd = Path(args.project_source_directory).absolute()
    args.output = [wd / output for output in args.output]
    args.input = [sd / filename for filename in args.input]
    args.include_dirs = args.include_dirs.split(sep)
    args.compiler_include_dirs = [
        d for d in args.compiler_include_dirs.split(sep) if d
    ]
    logger.debug(f"{args.declaration_names=}")
    logger.debug(f"{args.input=}")
    return args


if __name__ == "__main__":
    args = parse_arguments()

    def stream_with_line_breaks(stream):
        def write(string):
            stream.write(string.replace(">>", "> >"))
            stream.write("\n")

        def write_verbose(string):
            write(string)
            print(string.replace(">>", "> >"))

        if args.verbose:
            return write_verbose
        else:
            return write

    parsed_declarations = parse_headers(
        args.input,
        args.include_dirs,
        args.declaration_names,
        args.run_external_castxml,
        args.compiler_path,
        args.compiler_include_dirs,
    )

    try:
        global_ns = declarations.get_global_namespace(parsed_declarations)
        algo_namespace = (
            global_ns.namespace("kwiver").namespace("vital").namespace("algo")
        )
        algo_namespace.init_optimizer()
    except RuntimeError as ex:
        logger.error("Error extracting namespaces")
        logger.error(ex)
        exit(1)

    sd = Path(args.project_source_directory).absolute()
    for filename, declaration, output in zip(
        args.input, args.declaration_names, args.output
    ):
        if not filename.exists():
            logger.error(f"{filename} does not exist")
            continue
        parse_file(
            algo_namespace=algo_namespace,
            filename=str(filename),
            project_source_directory=sd,
            output_basename=str(output),
            handwritten_trampoline_directory=(
                args.handwritten_trampoline_directory),
        )
