"""Auto-generation engine: inspect MIDAS ABC subclasses → NodeTypeSpec."""
from __future__ import annotations

import importlib
import inspect
import types
import typing
from abc import ABC

from numpy import ndarray

from midas_gui.session import (
    NodeTypeSpec,
    PortSpec,
    PortDirection,
    PortType,
)

# ── ABC → output port / category mapping ─────────────────────────────────

# Lazy imports: these are resolved at discovery time so the module can be
# imported without pulling in all of MIDAS.  The mapping is:
#   (module_path, class_name) → (output_port_name, PortType, category)
# Endpoint ABCs (no output port) have output_port_name=None.

_ABC_REGISTRY: list[tuple[str, str, str | None, PortType | None, str]] = [
    ("midas.models.fields",           "FieldModel",         "field",            PortType.FIELD,            "Field Models"),
    ("midas.models.diagnostics",      "DiagnosticModel",    "diagnostic_model", PortType.DIAGNOSTIC_MODEL, "Diagnostic Models"),
    ("midas.likelihoods",             "LikelihoodFunction", "likelihood",       PortType.LIKELIHOOD,       "Likelihoods"),
    ("midas.likelihoods.uncertainties","UncertaintyModel",  "uncertainties",    PortType.UNCERTAINTIES,     "Uncertainty Models"),
    ("midas.state",                   "BasePrior",          None,               None,                      "Priors"),
]

# ── Type annotation → PortType mapping ───────────────────────────────────

# Lazy-resolved: maps Python types to the PortType they correspond to.
# Built once by _ensure_type_map().

_TYPE_MAP: dict[type, PortType] | None = None


def _ensure_type_map():
    global _TYPE_MAP
    if _TYPE_MAP is not None:
        return
    from midas.parameters import ParameterVector, FieldRequest, Coordinates
    from midas.likelihoods.uncertainties import UncertaintyModel

    _TYPE_MAP = {
        ndarray:           PortType.ARRAY,
        ParameterVector:   PortType.PARAMS,
        FieldRequest:      PortType.FIELD_REQUEST,
        UncertaintyModel:  PortType.UNCERTAINTIES,
    }
    # Coordinates is a type alias (dict[str, ndarray]).  We store the
    # *alias object* so we can match it during annotation inspection.
    _TYPE_MAP[Coordinates] = PortType.COORDINATES


# Config-property types: these become properties panel widgets, not ports.
_CONFIG_TYPES: set[type] = {str, int, float, tuple}


def _resolve_abc_classes() -> list[tuple[type, str | None, PortType | None, str]]:
    """Import and return (abc_class, output_name, output_type, category)."""
    result = []
    for mod_path, cls_name, out_name, out_type, category in _ABC_REGISTRY:
        mod = importlib.import_module(mod_path)
        abc_cls = getattr(mod, cls_name)
        result.append((abc_cls, out_name, out_type, category))
    return result


# ── Annotation helpers ───────────────────────────────────────────────────

def _get_type_hints(cls: type) -> dict[str, typing.Any]:
    """Get type hints for cls.__init__, resolving forward refs."""
    try:
        return typing.get_type_hints(cls.__init__)
    except Exception:
        return {}


def _is_coordinates_annotation(ann) -> bool:
    """Check if annotation is the Coordinates type alias (dict[str, ndarray])."""
    from midas.parameters import Coordinates
    # Python 3.12+ type alias: compare directly
    if ann is Coordinates:
        return True
    # Also match typing.TypeAliasType value
    if hasattr(ann, '__value__'):
        return ann.__value__ is Coordinates or ann is Coordinates
    # Fallback: check if it's dict[str, ndarray] structurally
    origin = typing.get_origin(ann)
    if origin is dict:
        args = typing.get_args(ann)
        if len(args) == 2 and args[0] is str and args[1] is ndarray:
            return True
    return False


def _classify_annotation(ann) -> tuple[str, tuple[PortType, ...] | type | None]:
    """Classify a single type annotation.

    Returns:
        ("port", tuple_of_port_types)  – should become an input port
        ("config", python_type)        – should become a config property
        ("unresolved", None)           – unknown type → variable-name field
    """
    _ensure_type_map()

    # Check Coordinates alias first (before dict generic check)
    if _is_coordinates_annotation(ann):
        return ("port", (PortType.COORDINATES,))

    # Direct match against known port types
    if ann in _TYPE_MAP:
        return ("port", (_TYPE_MAP[ann],))

    # Union types: X | Y
    origin = typing.get_origin(ann)
    if origin is types.UnionType or origin is typing.Union:
        args = typing.get_args(ann)
        port_types = []
        for arg in args:
            if arg is type(None):
                continue  # Optional marker
            if arg in _TYPE_MAP:
                port_types.append(_TYPE_MAP[arg])
        if port_types:
            return ("port", tuple(port_types))

    # Config property types
    if ann in _CONFIG_TYPES:
        return ("config", ann)

    # Tuple subtypes (e.g. tuple[float, float])
    if origin is tuple:
        return ("config", tuple)

    return ("unresolved", None)


# ── Main introspection ───────────────────────────────────────────────────

def generate_node_spec(
    cls: type,
    abc_info: tuple[type, str | None, PortType | None, str],
) -> NodeTypeSpec | None:
    """Inspect a single concrete class and produce a NodeTypeSpec.

    Parameters
    ----------
    cls
        The concrete MIDAS class to introspect.
    abc_info
        (abc_class, output_port_name_or_None, output_PortType_or_None, category)

    Returns
    -------
    NodeTypeSpec or None if the class cannot be introspected.
    """
    abc_cls, out_port_name, out_port_type, category = abc_info

    # Skip abstract classes
    if inspect.isabstract(cls):
        return None

    hints = _get_type_hints(cls)
    sig = inspect.signature(cls.__init__)

    input_ports: list[PortSpec] = []
    default_properties: dict[str, typing.Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        ann = hints.get(param_name, inspect.Parameter.empty)
        if ann is inspect.Parameter.empty:
            # No type annotation — treat as unresolved
            default_properties[param_name] = ""
            continue

        has_default = param.default is not inspect.Parameter.empty
        is_optional = has_default and param.default is None

        role, detail = _classify_annotation(ann)

        if role == "port":
            port_types = detail
            required = not is_optional
            input_ports.append(
                PortSpec(param_name, port_types, PortDirection.INPUT, required)
            )
        elif role == "config":
            default_val = param.default if has_default else _config_default(detail)
            default_properties[param_name] = default_val
        else:
            # Unresolved type → variable-name field in properties panel.
            # Store "" as the variable name; track metadata separately.
            default_properties[param_name] = ""
            type_name = getattr(ann, "__name__", None) or getattr(ann, "__qualname__", None) or str(ann)
            unresolved_entry = {"type_name": type_name, "has_default": has_default}
            if has_default:
                # Build a constructor expression for codegen (e.g. "SquaredExponential()")
                default_cls = type(param.default)
                default_repr = repr(param.default)
                # If repr looks like an object address, fall back to ClassName()
                if default_repr.startswith("<"):
                    default_repr = f"{default_cls.__name__}()"
                unresolved_entry["default_repr"] = default_repr
                unresolved_entry["default_module"] = default_cls.__module__
                unresolved_entry["default_class"] = default_cls.__qualname__
            default_properties.setdefault("_unresolved", {})[param_name] = unresolved_entry

    # Build output ports
    output_ports: tuple[PortSpec, ...] = ()
    if out_port_name is not None and out_port_type is not None:
        output_ports = (
            PortSpec(out_port_name, (out_port_type,), PortDirection.OUTPUT),
        )

    # Store the originating class reference and module for codegen
    default_properties["_class"] = f"{cls.__module__}.{cls.__qualname__}"

    return NodeTypeSpec(
        type_id=cls.__name__,
        display_name=cls.__name__,
        category=category,
        input_ports=tuple(input_ports),
        output_ports=output_ports,
        default_properties=default_properties,
    )


def _config_default(python_type: type):
    """Sensible default for a config property type."""
    if python_type is str:
        return ""
    if python_type is int:
        return 0
    if python_type is float:
        return 0.0
    if python_type is tuple:
        return (0.0, 1.0)
    return ""


# ── Discovery ────────────────────────────────────────────────────────────

# Modules to scan for built-in MIDAS classes
_BUILTIN_MODULES = [
    "midas.models.fields",
    "midas.models.diagnostics",
    "midas.likelihoods",
    "midas.likelihoods.uncertainties",
    "midas.priors",
    "midas.state",
]


def discover_builtin_nodes() -> dict[str, NodeTypeSpec]:
    """Scan built-in MIDAS modules and return auto-generated NodeTypeSpecs.

    Returns a dict mapping type_id → NodeTypeSpec for every concrete
    subclass of a supported ABC found in the built-in modules.
    """
    abc_infos = _resolve_abc_classes()
    specs: dict[str, NodeTypeSpec] = {}

    for mod_path in _BUILTIN_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if not isinstance(obj, type):
                continue

            for abc_cls, out_name, out_type, category in abc_infos:
                if obj is abc_cls:
                    continue  # skip the ABC itself
                if not issubclass(obj, abc_cls):
                    continue
                if inspect.isabstract(obj):
                    continue
                if obj.__name__ in specs:
                    continue  # already registered (e.g. re-exported)

                spec = generate_node_spec(obj, (abc_cls, out_name, out_type, category))
                if spec is not None:
                    specs[spec.type_id] = spec
                break  # matched an ABC, don't check others

    return specs


def discover_user_module(file_path: str) -> dict[str, NodeTypeSpec]:
    """Import a user .py file and return NodeTypeSpecs for any ABC subclasses.

    Parameters
    ----------
    file_path
        Absolute path to the .py file to import.

    Returns
    -------
    dict mapping type_id → NodeTypeSpec for discovered classes.
    """
    import sys
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return {}

    # Temporarily add the file's directory to sys.path
    parent = str(path.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)

    try:
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            return {}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return {}
    finally:
        if added:
            sys.path.remove(parent)

    abc_infos = _resolve_abc_classes()
    specs: dict[str, NodeTypeSpec] = {}

    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if not isinstance(obj, type):
            continue
        for abc_cls, out_name, out_type, category in abc_infos:
            if obj is abc_cls:
                continue
            if not issubclass(obj, abc_cls):
                continue
            if inspect.isabstract(obj):
                continue

            node_spec = generate_node_spec(obj, (abc_cls, out_name, out_type, category))
            if node_spec is not None:
                specs[node_spec.type_id] = node_spec
            break

    return specs
