import contextlib
import io
import json

import numpy as np


class BoundedOutput(io.StringIO):
    def __init__(self, stream, emit=None):
        super().__init__()
        self.stream = stream
        self.emit = emit
        self.truncated = False

    def write(self, text):
        available = max(0, 8000 - self.tell())
        chunk = text[:available]
        super().write(chunk)
        if self.emit and chunk:
            self.emit(self.stream, chunk)
        if len(text) > available and not self.truncated:
            self.truncated = True
            if self.emit:
                self.emit(self.stream, "\n[Output truncated at 8,000 characters]\n")
        return len(text)


def validate_label(name, context):
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ValueError(f"{context} labels must be nonempty strings of at most 80 characters")


def export_group(namespace, group):
    exports = namespace.get(group)
    if not isinstance(exports, dict):
        raise ValueError(f"Define {group} as a dictionary of display labels to numeric values; use {{}} for an empty group")
    if len(exports) > 32:
        raise ValueError(f"{group} supports at most 32 displayed values")
    variables = []
    for name, value in exports.items():
        validate_label(name, group)
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or array.ndim > 2:
            raise ValueError(f"{group}[{name!r}] must be a real numeric scalar, vector, or matrix; got {array.dtype}, shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{group}[{name!r}] must contain only finite values")
        if array.ndim == 0:
            preview = array.reshape(1, 1)
        elif array.ndim == 1:
            preview = array[:12].reshape(1, -1)
        else:
            preview = array[:12, :12]
        variables.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype),
                          "values": preview.tolist(), "size": int(array.size),
                          "truncated": preview.size != array.size})
    return variables


def export_plots(namespace, cache):
    plots = namespace.get("PLOTS", {})
    if not isinstance(plots, dict) or len(plots) > 12:
        raise ValueError("PLOTS must be a dictionary with at most 12 plots")
    result = []
    total = 0
    retained = set()
    for title, series in plots.items():
        validate_label(title, "PLOTS")
        if not isinstance(series, dict) or not 1 <= len(series) <= 8:
            raise ValueError(f"PLOTS[{title!r}] must contain 1 to 8 named series")
        curves = []
        for name, values in series.items():
            validate_label(name, "Plot series")
            shaded = None
            if isinstance(values, dict):
                if set(values) != {"points", "area"}:
                    raise ValueError("Shaded series require points and area")
                region = values["area"]
                if not isinstance(region, dict) or set(region) != {"lower", "upper", "value"}:
                    raise ValueError("Plot area requires lower, upper, and value")
                bounds = np.asarray([region["lower"], region["upper"], region["value"]])
                if bounds.shape != (3,) or bounds.dtype.kind not in "iuf" or not np.isfinite(bounds[:2]).all():
                    raise ValueError("Plot area requires real scalar values and finite bounds")
                shaded = {"lower": float(bounds[0]), "upper": float(bounds[1]),
                          "value": float(bounds[2]) if np.isfinite(bounds[2]) else None}
                values = values["points"]
            array = np.asarray(values)
            if array.dtype.kind not in "iuf" or array.ndim != 2 or array.shape[1] != 2 or not 2 <= len(array) <= 4097:
                raise ValueError(f"Plot series {name!r} must be a real numeric (N, 2) array with 2 to 4097 points")
            if not np.isfinite(array[:, 0]).all():
                raise ValueError(f"Plot series {name!r} must have finite x coordinates")
            total += len(array)
            if total > 100000:
                raise ValueError("PLOTS supports at most 100000 points in total")
            key = (title, name)
            signature = (array.dtype.str, array.shape, array.tobytes())
            cached = cache.get(key)
            if cached is not None and cached[0] == signature:
                points = cached[1]
            else:
                points = array.astype(np.float64).tolist()
                for index in np.flatnonzero(~np.isfinite(array[:, 1])):
                    points[index][1] = None
            cache[key] = (signature, points)
            retained.add(key)
            curve = {"name": name, "points": points}
            if shaded is not None:
                curve["area"] = shaded
            curves.append(curve)
        result.append({"title": title, "series": curves})
    for key in set(cache) - retained:
        del cache[key]
    return result


def plot_updates(previous, current):
    structure = lambda plots: [(plot["title"], [series["name"] for series in plot["series"]]) for plot in plots]
    if previous is None or structure(previous) != structure(current):
        return None
    updates = []
    for old_plot, plot in zip(previous, current):
        changes = []
        for old_series, series in zip(old_plot["series"], plot["series"]):
            if old_series["points"] is series["points"] and old_series.get("area") == series.get("area"):
                continue
            change = {"name": series["name"], "area": series.get("area")}
            if old_series["points"] is not series["points"]:
                change["points"] = series["points"]
            changes.append(change)
        if changes:
            updates.append({"title": plot["title"], "series": changes})
    return updates


class LessonSession:
    def __init__(self):
        self.namespace = None
        self.revision = None
        self.parameters = {}
        self.code = None
        self.compiled = None
        self.plot_cache = {}
        self.plots = None

    def run(self, request):
        try:
            if "code" not in request:
                if self.namespace is None or request.get("revision") != self.revision:
                    raise ValueError("Lesson session expired; run the full code again")
                parameters = request.get("parameters")
                if not isinstance(parameters, dict) or not parameters or not parameters.keys() <= self.parameters.keys():
                    raise ValueError("Updates must contain declared PARAMETERS")
                parameters = {**self.parameters, **parameters}
                self.validate_parameters(parameters)
                exports = self.namespace["update"](**parameters)
                if not isinstance(exports, dict) or not {"INPUTS", "OUTPUTS"} <= exports.keys() or not exports.keys() <= {"INPUTS", "OUTPUTS", "PLOTS"}:
                    raise ValueError("update must return INPUTS and OUTPUTS, with optional PLOTS")
                self.namespace.update(exports)
            else:
                self.plot_cache = {}
                self.plots = None
                namespace = {"__name__": "__lesson__"}
                if request["code"] != self.code:
                    self.compiled = compile(request["code"], "lesson.py", "exec")
                    self.code = request["code"]
                exec(self.compiled, namespace)
                parameters = namespace.get("PARAMETERS", {})
                self.validate_parameters(parameters)
                if parameters and not callable(namespace.get("update")):
                    raise ValueError("PARAMETERS requires an update function")
                self.namespace = namespace
                self.revision = request.get("revision")
            self.parameters = dict(parameters)
            response = {"inputs": export_group(self.namespace, "INPUTS"),
                        "outputs": export_group(self.namespace, "OUTPUTS"),
                        "parameters": self.parameters}
            if "code" in request or "PLOTS" in exports:
                plots = export_plots(self.namespace, self.plot_cache)
                updates = plot_updates(self.plots, plots)
                if updates is None:
                    response["plots"] = plots
                elif updates:
                    response["plotUpdates"] = updates
                self.plots = plots
            return response
        except Exception:
            self.namespace = None
            self.parameters = {}
            self.plot_cache = {}
            self.plots = None
            raise

    @staticmethod
    def validate_parameters(parameters):
        if not isinstance(parameters, dict) or len(parameters) > 32:
            raise ValueError("PARAMETERS must be a dictionary with at most 32 values")
        for name, value in parameters.items():
            if not isinstance(name, str) or not name.isidentifier() or type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError("PARAMETERS must map Python names to finite numeric scalars")


session = LessonSession()


def run_request(payload, emit_output=None):
    captured_stdout = BoundedOutput("stdout", emit_output)
    captured_stderr = BoundedOutput("stderr", emit_output)
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        response = session.run(json.loads(payload))
    response["stdout"] = captured_stdout.getvalue()
    response["stderr"] = captured_stderr.getvalue()
    return json.dumps(response, allow_nan=False)