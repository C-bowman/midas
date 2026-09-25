from collections.abc import Callable
from dataclasses import dataclass
from numpy import isfinite, ndarray, zeros
from numpy.random import default_rng
from numpy.linalg import norm as l2_norm
from midas.state import PlasmaState
from midas.posterior import NormalisedCost


def finite_difference_gradient(
    func: Callable[[ndarray], float],
    x0: ndarray,
    param_slice: slice,
    step: float,
) -> ndarray:

    indices = range(*param_slice.indices(x0.size))
    numerical = zeros(len(indices))
    for output_index, index in enumerate(indices):
        x_neg, x_pos = x0.copy(), x0.copy()
        delta = step * max(1.0, abs(x0[index]))
        x_neg[index] -= delta
        x_pos[index] += delta
        f_neg = func(x_neg)
        f_pos = func(x_pos)
        numerical[output_index] = (f_pos - f_neg) / (2 * delta)
    return numerical


def vector_error_metrics(u: ndarray, v: ndarray) -> tuple[float, float]:
    mag_u = l2_norm(u)
    mag_v = l2_norm(v)

    if mag_u == 0 and mag_v == 0:
        return 0.0, 0.0
    if (mag_u == 0) != (mag_v == 0):
        return 1.0, 1.0
    
    direction_error = l2_norm(u / mag_u - v / mag_v)
    magnitude_error = abs(mag_u - mag_v) / max(mag_u, mag_v)
    return magnitude_error, direction_error


def estimate_gradient_errors(
    func: Callable[[ndarray], float],
    x0: ndarray,
    analytic: ndarray,
    param_slice: slice,
    initial_step: float,
    max_iterations: int = 10
) -> tuple[float, float]:
    numerical = finite_difference_gradient(func, x0, param_slice, initial_step)
    mag_err, dir_err = vector_error_metrics(analytic, numerical)

    for _ in range(max_iterations):
        initial_step /= 2
        numerical = finite_difference_gradient(func, x0, param_slice, initial_step)
        new_mag_err, new_dir_err = vector_error_metrics(analytic, numerical)
        if (new_mag_err > mag_err) or (new_dir_err > dir_err):
            return mag_err, dir_err

        mag_err, dir_err = new_mag_err, new_dir_err
    return mag_err, dir_err


ANSI_COLORS = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}


def _colorize(text: str, color: str | int) -> str:
    if isinstance(color, str):
        try:
            ansi_code = ANSI_COLORS[color.lower()]
        except KeyError as error:
            available = ", ".join(ANSI_COLORS)
            raise ValueError(
                f"Unknown ANSI color {color!r}; expected one of: {available}"
            ) from error
    else:
        ansi_code = color
    return f"\033[{ansi_code}m{text}\033[0m"


@dataclass
class GradientReport:
    """
    Summarise the results of validating posterior component gradients.

    Instances are returned by :func:`validate_gradient`; users do not normally
    construct reports directly.

    A report evaluates as true when every sampled component/parameter gradient
    passed both error tolerances. Use :meth:`print_report` for a readable table or
    inspect :attr:`results` programmatically.

    :ivar success: \
        Whether every gradient check passed.

    :ivar results: \
        Nested mapping from component names to parameter names and their maximum
        direction error, maximum magnitude error, and number of passing samples.

    :ivar n_samples: \
        Number of validation points sampled for each gradient.
    """

    success: bool
    results: dict
    n_samples: int

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        return self.format()

    def print_report(self, *, color: bool = True) -> None:
        """Print the formatted validation report.

        :param color: Whether to color statuses and component headings with ANSI
            escape sequences.
        """
        print(self.format(color=color))

    def format(self, *, color: bool = False) -> str:
        """Format the validation results as a readable table.

        :param color: Whether to color statuses and component headings with ANSI
            escape sequences.
        :return: The formatted validation report.
        """
        status_color = {"PASS": "green", "FAIL": "red"}
        overall_status = "PASS" if self.success else "FAIL"
        if color:
            overall_status = _colorize(
                overall_status, status_color[overall_status]
            )
        lines = [f"Gradient validation: {overall_status}"]
        headers = (
            "Parameter",
            "Status",
            "Max direction error",
            "Max magnitude error",
            "Pass rate",
        )
        for component, parameters in self.results.items():
            rows = []
            for name, accuracy in parameters.items():
                passed = accuracy["n_passes"] == self.n_samples and self.n_samples > 0
                pass_rate = (
                    f"{100 * accuracy['n_passes'] / self.n_samples:.1f}%"
                    if self.n_samples > 0 else "N/A"
                )
                rows.append((
                    name,
                    "PASS" if passed else "FAIL",
                    f"{accuracy['max_dir_err']:.3g}",
                    f"{accuracy['max_mag_err']:.3g}",
                    pass_rate,
                ))

            widths = [
                max([len(header), *(len(row[index]) for row in rows)])
                for index, header in enumerate(headers)
            ]
            component_header = f"[{component}]"
            if color:
                component_header = _colorize(component_header, "cyan")
            lines.extend([
                "",
                component_header,
                "\t" + " | ".join(
                    header.ljust(width) for header, width in zip(headers, widths)
                ),
                "\t" + "-+-".join("-" * width for width in widths),
            ])
            for row in rows:
                status = row[1].ljust(widths[1])
                if color:
                    status = _colorize(
                        status, status_color[row[1]]
                    )
                lines.append(
                    "\t"
                    + " | ".join([
                        row[0].ljust(widths[0]),
                        status,
                        *(value.rjust(width) for value, width in zip(row[2:], widths[2:])),
                    ])
                )
        return "\n".join(lines)

    
def validate_gradient(
    parameter_bounds: ndarray,
    n_samples: int = 5,
    dir_tol: float = 1e-4,
    mag_tol: float = 1e-4,
    initial_step: float = 1e-6,
) -> GradientReport:
    """
    Validate every posterior component's analytic gradient calculation.

    Test points are sampled uniformly inside the central 80% of the parameter
    bounds after normalisation. For each component and parameter vector, its
    analytic gradient is compared with a centred finite-difference estimate. The
    finite-difference step is repeatedly halved until either error metric worsens
    or the refinement limit is reached.

    :param parameter_bounds: \
        Lower and upper bounds for every posterior parameter, as an array with
        shape ``(PlasmaState.n_params, 2)``.

    :param n_samples: \
        Number of random normalised parameter points to test.

    :param dir_tol: \
        Maximum accepted error between the gradient directions.

    :param mag_tol: \
        Maximum accepted relative error between gradient magnitudes.

    :param initial_step: \
        Largest relative finite-difference step considered. Subsequent candidates
        are generated by repeatedly halving this value.

    :return: \
        A report containing aggregate errors and pass counts for each posterior
        component and parameter vector.

    :raises ValueError: \
        If any sampled point produces a non-finite cost.
    """

    norm = NormalisedCost(bounds=parameter_bounds)
    rng = default_rng()

    test_points = rng.uniform(
        low=0.1, high=0.9, size=(n_samples, PlasmaState.n_params)
    )

    finite_costs = all(isfinite(norm.cost(p)) for p in test_points)
    if not finite_costs:
        raise ValueError("Some test points resulted in non-finite costs.")


    results = {
        c.name: {
            p: {"max_dir_err": 0.0, "max_mag_err": 0.0, "n_passes": 0}
            for p in PlasmaState.parameter_names
        }
        for c in PlasmaState.components
    }

    for point in test_points:
        for component in PlasmaState.components:
            analytic_grad = norm.component_cost_gradient(point, component.name)
            for parameter, slc in PlasmaState.slices.items():
                mag_err, dir_err = estimate_gradient_errors(
                    func=lambda x: norm.component_cost(x, component.name),
                    x0=point,
                    analytic=analytic_grad[slc],
                    param_slice=slc,
                    initial_step=initial_step,
                )

                r = results[component.name][parameter]
                r["max_dir_err"] = max(r["max_dir_err"], dir_err)
                r["max_mag_err"] = max(r["max_mag_err"], mag_err)
                r["n_passes"] += (dir_err < dir_tol and mag_err < mag_tol)

    success = all(
        r["n_passes"] == n_samples
        for parameters in results.values()
        for r in parameters.values()
    )
    return GradientReport(success, results, n_samples)