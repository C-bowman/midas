from __future__ import annotations
from textwrap import dedent
from midas_gui.session import GraphModel, NodeModel, Edge, NODE_TYPES


optimization_template = \
"""

# ── Optimization ──────────────────────────────────────────
from scipy.optimize import minimize
from midas import posterior

# initial guess for optimization
initial_guess = np.ones(PlasmaState.n_params)  # TODO: replace with an informed guess

# Build bounds for optimization and sampling
bounds = PlasmaState.build_bounds(
    parameter_bounds={p: (0.0, 10.0) for p in PlasmaState.parameter_set}  # TODO: specify informed bounds
)

opt_result = minimize(
    posterior.cost,
    x0=initial_guess,
    jac=posterior.cost_gradient,
    method='L-BFGS-B',
    bounds=bounds,
)
map_estimate_params = opt_result.x
"""

mcmc_template = \
"""

# ── MCMC Sampling ─────────────────────────────────────────
from inference.approx import conditional_moments
from inference.mcmc import HamiltonianChain

# Use conditional variance to estimate inverse-mass for the HMC sampler
_, conditional_variance = conditional_moments(
    posterior=posterior.log_probability,
    conditioning_point=map_estimate_params,
    bounds=[b for b in bounds],
)

chain = HamiltonianChain(
    posterior=posterior.log_probability,
    grad=posterior.gradient,
    start=map_estimate_params,
    inverse_mass=conditional_variance,
    bounds=bounds.T,
)
chain.advance(5000)
samples = chain.get_sample(burn=1000)
"""





def generate_script(
    graph: GraphModel,
    runnable: bool = False,
    imported_modules: list[str] | None = None,
) -> str:
    """Traverse the node graph and emit a valid MIDAS Python script."""
    lines: list[str] = []
    errors = graph.validate()

    if errors:
        lines.append("# WARNING: The graph has validation errors:")
        for err in errors:
            lines.append(f"#   - {err}")
        lines.append("")

    # Add imported module directories to sys.path
    if imported_modules:
        from pathlib import PurePosixPath
        lines.append("import sys")
        lines.append("# Add imported module directories to the path")
        seen_dirs: set[str] = set()
        for mod_path in imported_modules:
            # Normalise to forward slashes first so PurePosixPath handles
            # both Windows and POSIX paths consistently on any platform.
            normalised = mod_path.replace("\\", "/")
            parent = str(PurePosixPath(normalised).parent)
            if parent not in seen_dirs:
                seen_dirs.add(parent)
                lines.append(f'sys.path.insert(0, "{parent}")')
        lines.append("")

    # Build variable name map: node_id -> python variable name
    var_names: dict[str, str] = {}
    for node in graph.nodes.values():
        name = node.properties.get("name", "").strip()
        if not name:
            name = f"{node.type_id.lower()}_{node.id}"
        var_names[node.id] = _sanitize_var(name)

    # Deduplicate variable names
    seen: dict[str, int] = {}
    for nid, vname in var_names.items():
        if vname in seen:
            seen[vname] += 1
            var_names[nid] = f"{vname}_{seen[vname]}"
        else:
            seen[vname] = 0

    # Determine needed imports
    imports = _collect_imports(graph)
    lines.append("import numpy as np")
    for module, names in sorted(imports.items()):
        names_str = ", ".join(sorted(names))
        lines.append(f"from {module} import {names_str}")
    lines.append("")

    # Emit nodes grouped by type with locality-aware ordering
    emitted: set[str] = set()
    _emit_grouped_nodes(graph, var_names, lines, emitted)

    # Emit any orphan nodes (defined but not connected to the analysis chain)
    orphans = [n for n in graph.nodes.values() if n.id not in emitted]
    if orphans:
        lines.append("# ── Unused nodes ──────────────────────────────────────────")
        lines.append("")
        for node in orphans:
            code = _emit_node(node, var_names, graph)
            if code:
                lines.extend(code)
                lines.append("")

    # Build posterior call
    lines.append("# ── Build posterior ────────────────────────────────────────")
    lines.append("")

    diag_likelihoods = [
        var_names[n.id] for n in graph.nodes.values()
        if n.type_id == "DiagnosticLikelihood"
    ]
    priors = [
        var_names[n.id] for n in graph.nodes.values()
        if NODE_TYPES.get(n.type_id, None) and NODE_TYPES[n.type_id].category == "Priors"
    ]
    field_models = [
        var_names[n.id] for n in graph.nodes.values()
        if NODE_TYPES.get(n.type_id, None) and NODE_TYPES[n.type_id].category == "Field Models"
    ]

    if diag_likelihoods or priors or field_models:
        lines.extend(dedent(f"""\
            PlasmaState.build_posterior(
                diagnostics=[{', '.join(diag_likelihoods)}],
                priors=[{', '.join(priors)}],
                field_models=[{', '.join(field_models)}],
            )""").splitlines())

    if runnable:
        lines.extend(dedent(optimization_template).splitlines())
        lines.extend(dedent(mcmc_template).splitlines())

    return "\n".join(lines) + "\n"


def _sanitize_var(name: str) -> str:
    out = ""
    for c in name:
        if c.isalnum() or c == "_":
            out += c
        else:
            out += "_"
    if out and out[0].isdigit():
        out = "_" + out
    return out or "node"


def _collect_imports(graph: GraphModel) -> dict[str, set[str]]:
    imports: dict[str, set[str]] = {}

    def _add(module: str, name: str):
        imports.setdefault(module, set()).add(name)

    for node in graph.nodes.values():
        type_id = node.type_id
        spec = NODE_TYPES.get(type_id)
        if not spec:
            continue

        # Hard-coded utility nodes
        if type_id == "ParameterVector":
            _add("midas.parameters", "ParameterVector")
        elif type_id == "FieldRequest":
            _add("midas.parameters", "FieldRequest")
        elif type_id == "DiagnosticLikelihood":
            _add("midas.likelihoods", "DiagnosticLikelihood")
        elif type_id in ("Array", "Coordinates"):
            pass  # numpy arrays / dicts — no special import
        else:
            # Auto-generated: derive import from _class metadata
            class_path = spec.default_properties.get("_class", "")
            if class_path:
                parts = class_path.rsplit(".", 1)
                if len(parts) == 2:
                    _add(parts[0], parts[1])

        # Import for unresolved defaults that use "Use default"
        unresolved = spec.default_properties.get("_unresolved", {})
        for key, meta in unresolved.items():
            use_default_key = f"_use_default_{key}"
            if node.properties.get(use_default_key, meta.get("has_default", False)):
                default_module = meta.get("default_module", "")
                default_class = meta.get("default_class", "")
                if default_module and default_class:
                    _add(default_module, default_class)

    # Always need PlasmaState if there's anything to build
    if graph.nodes:
        _add("midas.state", "PlasmaState")

    return imports


def _emit_node(
    node: NodeModel,
    var_names: dict[str, str],
    graph: GraphModel,
) -> list[str]:
    var = var_names[node.id]
    props = node.properties
    lines: list[str] = []

    if node.type_id == "ParameterVector":
        name = props.get("name", var)
        size = props.get("size", 1)
        lines.append(f'{var} = ParameterVector(name="{name}", size={size})')

    elif node.type_id == "Array":
        config = props.get("values_config", {})
        source = config.get("source", "placeholder")
        if source == "placeholder":
            lines.append(f"{var}: np.ndarray  # TODO: assign array data")
        elif source == "linspace":
            lines.append(
                f'{var} = np.linspace({config.get("start", 0)}, '
                f'{config.get("stop", 1)}, {config.get("num", 10)})'
            )
        elif source == "arange":
            lines.append(
                f'{var} = np.arange({config.get("start", 0)}, '
                f'{config.get("stop", 1)}, {config.get("step", 0.1)})'
            )
        elif source == "constant":
            lines.append(
                f'{var} = np.full({config.get("size", 10)}, '
                f'{config.get("value", 0.0)})'
            )
        elif source == "file":
            path = config.get("path", "data.npy")
            npz_key = config.get("npz_key")
            if path.endswith(".csv"):
                lines.append(f'{var} = np.loadtxt("{path}", delimiter=",")')
            elif npz_key is not None:
                lines.append(f'{var} = np.load("{path}")["{npz_key}"]')
            else:
                lines.append(f'{var} = np.load("{path}")')
        else:
            lines.append(f"{var} = np.array([])  # TODO: specify data")

    elif node.type_id == "Coordinates":
        coord_names = props.get("coordinate_names", [])
        lines.append(f'{var} = {{')
        for cname in coord_names:
            edge = _find_input_edge(graph, node.id, cname)
            cvar = var_names[edge.source_node_id] if edge else f"None  # TODO: connect {cname}"
            lines.append(f'    "{cname}": {cvar},')
        lines.append(f'}}')

    elif node.type_id == "FieldRequest":
        field_edge = _find_input_edge(graph, node.id, "field")
        coord_edge = _find_input_edge(graph, node.id, "coordinates")
        coord_var = var_names[coord_edge.source_node_id] if coord_edge else "{}  # TODO"
        # The field name comes from the connected FieldModel's field_name property
        fr_name = var
        if field_edge:
            src_node = graph.nodes.get(field_edge.source_node_id)
            if src_node:
                fr_name = src_node.properties.get("field_name", "").strip() or var
        lines.extend(dedent(f"""\
            {var} = FieldRequest(
                name="{fr_name}",
                coordinates={coord_var},
            )""").splitlines())

    elif node.type_id == "DiagnosticLikelihood":
        name = props.get("name", var)
        model_edge = _find_input_edge(graph, node.id, "diagnostic_model")
        like_edge = _find_input_edge(graph, node.id, "likelihood")
        model_var = var_names[model_edge.source_node_id] if model_edge else "None  # TODO"
        like_var = var_names[like_edge.source_node_id] if like_edge else "None  # TODO"
        lines.extend(dedent(f"""\
            {var} = DiagnosticLikelihood(
                diagnostic_model={model_var},
                likelihood={like_var},
                name="{name}",
            )""").splitlines())

    else:
        # Generic emitter for all auto-generated nodes (ABC subclasses)
        lines.extend(_emit_auto_node(node, var, var_names, graph))

    return lines


def _emit_auto_node(
    node: NodeModel,
    var: str,
    var_names: dict[str, str],
    graph: GraphModel,
) -> list[str]:
    """Emit constructor call for an auto-generated node."""
    lines: list[str] = []
    spec = node.spec
    props = node.properties
    unresolved_meta = props.get("_unresolved", spec.default_properties.get("_unresolved", {}))

    # Build lookups
    port_by_name = {p.name: p for p in spec.input_ports}
    param_order = spec.default_properties.get("_param_order", [])

    # Determine argument order: use _param_order if available,
    # otherwise fall back to ports-then-properties order.
    if param_order:
        ordered_params = param_order
    else:
        ordered_params = (
            [p.name for p in spec.input_ports]
            + [k for k in spec.default_properties if k not in ("_class", "_unresolved", "_param_order", "name")]
        )

    # Check if "name" is a real __init__ parameter
    has_name_param = "name" in spec.default_properties and "name" not in ("_class", "_unresolved", "_param_order")

    class_name = node.type_id
    if has_name_param:
        name_val = props.get("name", var)
        lines.append(f'{var} = {class_name}(')
        lines.append(f'    name="{name_val}",')
    else:
        lines.append(f'{var} = {class_name}(')

    # Emit arguments in order
    for param_name in ordered_params:
        if param_name == "name":
            continue  # already emitted above

        if param_name in port_by_name:
            port = port_by_name[param_name]
            edge = _find_input_edge(graph, node.id, port.name)
            if edge:
                lines.append(f"    {port.name}={var_names[edge.source_node_id]},")
            elif not port.required:
                pass  # skip optional unconnected ports
            else:
                lines.append(f"    {port.name}=None,  # TODO: connect {port.name}")

        elif param_name in unresolved_meta:
            meta = unresolved_meta[param_name]
            use_default_key = f"_use_default_{param_name}"
            if props.get(use_default_key, meta.get("has_default", False)):
                default_repr = meta.get("default_repr", "None")
                lines.append(f"    {param_name}={default_repr},")
            else:
                val = props.get(param_name, "")
                if val:
                    lines.append(f"    {param_name}={val},")
                else:
                    type_name = meta.get("type_name", "?")
                    lines.append(f"    # {param_name}: {type_name}  # TODO: assign value")

        elif param_name in spec.default_properties:
            default_val = spec.default_properties[param_name]
            if isinstance(default_val, str):
                val = props.get(param_name, default_val)
                lines.append(f'    {param_name}="{val}",')
            elif isinstance(default_val, (int, float)):
                val = props.get(param_name, default_val)
                lines.append(f"    {param_name}={val},")
            elif isinstance(default_val, tuple):
                val = props.get(param_name, default_val)
                lines.append(f"    {param_name}={val},")

    lines.append(f')')
    return lines


# Type groups for locality-aware emission ordering
_TYPE_GROUPS = [
    ("Field Models",            "# ── Field models ─────────────────────────────────────────"),
    ("Diagnostic Models",       "# ── Diagnostic models ────────────────────────────────────"),
    ("Likelihoods",             "# ── Likelihood models ────────────────────────────────────"),
    ("DiagnosticLikelihood",    "# ── Diagnostic likelihoods ───────────────────────────────"),
    ("Priors",                  "# ── Priors ───────────────────────────────────────────────"),
]


def _emit_grouped_nodes(
    graph: GraphModel,
    var_names: dict[str, str],
    lines: list[str],
    emitted: set[str],
):
    """Emit nodes grouped by type, with upstream dependencies inlined."""
    for group_key, header in _TYPE_GROUPS:
        # Collect nodes for this group
        if group_key == "DiagnosticLikelihood":
            group_nodes = [n for n in graph.nodes.values() if n.type_id == "DiagnosticLikelihood"]
        elif group_key == "Likelihoods":
            # Likelihoods category but excluding DiagnosticLikelihood
            group_nodes = [
                n for n in graph.nodes.values()
                if n.type_id != "DiagnosticLikelihood"
                and NODE_TYPES.get(n.type_id) and NODE_TYPES[n.type_id].category == "Likelihoods"
            ]
        else:
            group_nodes = [
                n for n in graph.nodes.values()
                if NODE_TYPES.get(n.type_id) and NODE_TYPES[n.type_id].category == group_key
            ]
        # Also include UncertaintyModel nodes in the Likelihoods group
        if group_key == "Likelihoods":
            group_nodes.extend(
                n for n in graph.nodes.values()
                if NODE_TYPES.get(n.type_id) and NODE_TYPES[n.type_id].category == "Uncertainty Models"
            )

        if not group_nodes:
            continue

        lines.append(header)
        lines.append("")

        for node in group_nodes:
            # Emit all upstream dependencies first
            upstream = _upstream_topo_order(graph, node.id)
            for dep in upstream:
                if dep.id not in emitted:
                    code = _emit_node(dep, var_names, graph)
                    if code:
                        lines.extend(code)
                        lines.append("")
                    emitted.add(dep.id)

            # Emit the node itself
            if node.id not in emitted:
                code = _emit_node(node, var_names, graph)
                if code:
                    lines.extend(code)
                    lines.append("")
                emitted.add(node.id)


def _upstream_topo_order(graph: GraphModel, node_id: str) -> list[NodeModel]:
    """Return all upstream nodes of *node_id* in topological order."""
    # Collect all upstream node IDs via BFS
    upstream_ids: set[str] = set()
    queue = [node_id]
    while queue:
        nid = queue.pop(0)
        for edge in graph.edges:
            if edge.target_node_id == nid and edge.source_node_id not in upstream_ids:
                upstream_ids.add(edge.source_node_id)
                queue.append(edge.source_node_id)

    # Build a sub-graph and topologically sort it
    sub_deps: dict[str, set[str]] = {nid: set() for nid in upstream_ids}
    for edge in graph.edges:
        if edge.target_node_id in upstream_ids and edge.source_node_id in upstream_ids:
            sub_deps[edge.target_node_id].add(edge.source_node_id)

    ordered: list[NodeModel] = []
    in_degree = {nid: len(deps) for nid, deps in sub_deps.items()}
    ready = sorted(nid for nid, deg in in_degree.items() if deg == 0)

    while ready:
        nid = ready.pop(0)
        ordered.append(graph.nodes[nid])
        for other_nid, deps in sub_deps.items():
            if nid in deps:
                in_degree[other_nid] -= 1
                if in_degree[other_nid] == 0:
                    ready.append(other_nid)
                    ready.sort()

    return ordered


def _find_input_edge(graph: GraphModel, node_id: str, port_name: str) -> Edge | None:
    for edge in graph.edges:
        if edge.target_node_id == node_id and edge.target_port_name == port_name:
            return edge
    return None
