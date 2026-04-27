from __future__ import annotations
from textwrap import dedent
from midas_gui.session import GraphModel, NodeModel, Edge, NODE_TYPES

# Hard-coded utility node type_ids — everything else uses generic emitter
_HARDCODED_TYPES = {"ParameterVector", "Array", "Coordinates", "FieldRequest", "DiagnosticLikelihood"}

# Category ordering for dependency-safe emission
_CATEGORY_ORDER = [
    "Data & Inputs",
    "Parameters & Fields",
    "Field Models",
    "Diagnostic Models",
    "Uncertainty Models",
    "Likelihoods",
    "Priors",
]


def generate_script(
    graph: GraphModel,
    runnable: bool = False,
    comments: bool = False,
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
        from pathlib import PurePath
        lines.append("import sys")
        lines.append("# Add imported module directories to the path")
        seen_dirs: set[str] = set()
        for mod_path in imported_modules:
            parent = str(PurePath(mod_path).parent)
            # Normalise to forward slashes for cross-platform scripts
            parent = parent.replace("\\", "/")
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

    if comments:
        lines.append("# ── Analysis construction ──────────────────────────────────")
        lines.append("")

    # Emit in dependency order: group by category, hard-coded order first
    ordered_nodes = _dependency_order(graph)

    for node in ordered_nodes:
        code = _emit_node(node, var_names, graph, comments)
        if code:
            lines.extend(code)
            lines.append("")

    # Build posterior call
    if comments:
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
        lines.extend(dedent(
            """

            # ── Optimization ──────────────────────────────────────────
            from scipy.optimize import minimize
            from midas import posterior

            # initial guess for optimization
            theta0 = np.ones(PlasmaState.n_params)

            # You can build bounds here using PlasmaState.build_bounds()
            bounds = None

            result = minimize(
                posterior.cost,
                x0=theta0,
                jac=posterior.cost_gradient,
                method='L-BFGS-B',
                bounds=bounds,
            )
            theta_map = result.x
            """).splitlines())
        lines.extend(dedent(
            """
            
            # ── MCMC Sampling ─────────────────────────────────────────
            from inference.mcmc import HamiltonianChain
            
            chain = HamiltonianChain(
                posterior=posterior.log_probability,
                grad=posterior.gradient,
                start=theta_map,
            )
            chain.advance(5000)
            samples = chain.get_sample(burn=1000)
            """).splitlines())

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
    comments: bool,
) -> list[str]:
    var = var_names[node.id]
    props = node.properties
    lines: list[str] = []

    if comments:
        lines.append(f"# {node.spec.display_name}: {props.get('name', var)}")

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
        elif source == "file":
            path = config.get("path", "data.npy")
            if path.endswith(".csv"):
                lines.append(f'{var} = np.loadtxt("{path}", delimiter=",")')
            else:
                lines.append(f'{var} = np.load("{path}")')
        else:
            lines.append(f"{var} = np.array([])  # TODO: specify data")

    elif node.type_id in ("PiecewiseLinearField", "CubicSplineField"):
        field_name = props.get("field_name", "").strip() or var
        axis_name = props.get("axis_name", "psi")
        axis_edge = _find_input_edge(graph, node.id, "axis")
        axis_var = var_names[axis_edge.source_node_id] if axis_edge else "np.linspace(0, 1, 10)  # TODO: connect axis"
        lines.extend(dedent(f"""\
            {var} = {node.type_id}(
                field_name="{field_name}",
                axis={axis_var},
                axis_name="{axis_name}",
            )""").splitlines())

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
        # FieldRequest needs the field name and coordinates
        field_var = var_names[field_edge.source_node_id] if field_edge else "None  # TODO"
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

    elif node.type_id == "LinearDiagnosticModel":
        field_edge = _find_input_edge(graph, node.id, "field")
        field_var = var_names[field_edge.source_node_id] if field_edge else "None  # TODO: connect field_request"
        matrix_edge = _find_input_edge(graph, node.id, "model_matrix")
        matrix_var = var_names[matrix_edge.source_node_id] if matrix_edge else "np.eye(10)  # TODO: connect model_matrix"
        lines.extend(dedent(f"""\
            {var} = LinearDiagnosticModel(
                field={field_var},
                model_matrix={matrix_var},
            )""").splitlines())

    elif node.type_id == "ConstantUncertainty":
        n_data = props.get("n_data", 1)
        param_name = props.get("parameter_name", "").strip() or f"{var}_sigma"
        lines.append(f'{var} = ConstantUncertainty(n_data={n_data}, parameter_name="{param_name}")')

    elif node.type_id == "GaussianLikelihood":
        name = props.get("name", var)
        data_edge = _find_input_edge(graph, node.id, "y_data")
        data_var = var_names[data_edge.source_node_id] if data_edge else "None  # TODO: connect data"
        sigma_edge = _find_input_edge(graph, node.id, "sigma")
        if sigma_edge:
            sigma_var = var_names[sigma_edge.source_node_id]
            lines.append(f'{var} = GaussianLikelihood(y_data={data_var}, sigma={sigma_var})')
        else:
            lines.append(f'{var} = GaussianLikelihood(y_data={data_var})')

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

    elif node.type_id == "GaussianPrior":
        name = props.get("name", var)
        mean_edge = _find_input_edge(graph, node.id, "mean")
        std_edge = _find_input_edge(graph, node.id, "standard_deviation")
        fr_edge = _find_input_edge(graph, node.id, "field_request")
        pv_edge = _find_input_edge(graph, node.id, "parameter_vector")

        mean_var = var_names[mean_edge.source_node_id] if mean_edge else "np.zeros(1)  # TODO"
        std_var = var_names[std_edge.source_node_id] if std_edge else "np.ones(1)  # TODO"

        lines.append(f'{var} = GaussianPrior(')
        lines.append(f'    name="{name}",')
        lines.append(f'    mean={mean_var},')
        lines.append(f'    standard_deviation={std_var},')
        if fr_edge:
            lines.append(f'    field_request={var_names[fr_edge.source_node_id]},')
        elif pv_edge:
            lines.append(f'    parameter_vector={var_names[pv_edge.source_node_id]},')
        else:
            lines.append('    # TODO: connect field_request or parameter_vector')
        lines.append(')')

    else:
        # Generic emitter for auto-generated nodes
        lines.extend(_emit_auto_node(node, var, var_names, graph, comments))

    return lines


def _emit_auto_node(
    node: NodeModel,
    var: str,
    var_names: dict[str, str],
    graph: GraphModel,
    comments: bool,
) -> list[str]:
    """Emit constructor call for an auto-generated node."""
    lines: list[str] = []
    spec = node.spec
    props = node.properties
    unresolved_meta = props.get("_unresolved", spec.default_properties.get("_unresolved", {}))

    # Build argument list from the spec's input ports and config properties
    args: list[str] = []

    # Iterate over input ports (in spec order)
    for port in spec.input_ports:
        edge = _find_input_edge(graph, node.id, port.name)
        if edge:
            args.append(f"    {port.name}={var_names[edge.source_node_id]},")
        elif not port.required:
            pass  # skip optional unconnected ports
        else:
            args.append(f"    {port.name}=None,  # TODO: connect {port.name}")

    # Iterate over config properties (in spec order)
    for key, default_val in spec.default_properties.items():
        if key in ("_class", "_unresolved", "name"):
            continue

        # Check if this is an unresolved type
        if key in unresolved_meta:
            meta = unresolved_meta[key]
            use_default_key = f"_use_default_{key}"
            if props.get(use_default_key, meta.get("has_default", False)):
                # Use the class default — emit the repr
                default_repr = meta.get("default_repr", "None")
                args.append(f"    {key}={default_repr},")
            else:
                val = props.get(key, "")
                if val:
                    args.append(f"    {key}={val},")
                else:
                    type_name = meta.get("type_name", "?")
                    args.append(f"    # {key}: {type_name}  # TODO: assign value")
        elif isinstance(default_val, str):
            val = props.get(key, default_val)
            args.append(f'    {key}="{val}",')
        elif isinstance(default_val, (int, float)):
            val = props.get(key, default_val)
            args.append(f"    {key}={val},")
        elif isinstance(default_val, tuple):
            val = props.get(key, default_val)
            args.append(f"    {key}={val},")

    # Check if "name" is in the spec's init signature (not just our GUI name prop)
    # For prior classes and others, "name" is the first positional arg
    has_name_param = any(
        key == "name" and key not in ("_class", "_unresolved")
        for key in spec.default_properties
    )

    class_name = node.type_id
    if has_name_param:
        name_val = props.get("name", var)
        lines.append(f'{var} = {class_name}(')
        lines.append(f'    "{name_val}",')
    else:
        lines.append(f'{var} = {class_name}(')

    lines.extend(args)
    lines.append(f')')
    return lines


def _dependency_order(graph: GraphModel) -> list[NodeModel]:
    """Topological sort: each node appears after all nodes connected to its inputs."""
    # Build adjacency: for each node, which nodes must come before it?
    dependencies: dict[str, set[str]] = {nid: set() for nid in graph.nodes}
    for edge in graph.edges:
        if edge.target_node_id in dependencies:
            dependencies[edge.target_node_id].add(edge.source_node_id)

    # Kahn's algorithm
    ordered: list[NodeModel] = []
    in_degree = {nid: len(deps) for nid, deps in dependencies.items()}
    queue = [nid for nid, deg in in_degree.items() if deg == 0]

    while queue:
        queue.sort()  # deterministic order for nodes at same depth
        nid = queue.pop(0)
        ordered.append(graph.nodes[nid])
        for other_nid, deps in dependencies.items():
            if nid in deps:
                in_degree[other_nid] -= 1
                if in_degree[other_nid] == 0:
                    queue.append(other_nid)

    # Any remaining nodes (cycles) get appended at the end
    seen = {n.id for n in ordered}
    for node in graph.nodes.values():
        if node.id not in seen:
            ordered.append(node)

    return ordered


def _find_input_edge(graph: GraphModel, node_id: str, port_name: str) -> Edge | None:
    for edge in graph.edges:
        if edge.target_node_id == node_id and edge.target_port_name == port_name:
            return edge
    return None
