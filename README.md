# refresh

A standalone, browser-side book of mathematics and statistics. The contents page
lists ordered chapters, with unpublished chapters marked Planned rather than
linked to empty pages. Mathematical Foundations opens the book with expandable
reminders from arithmetic through graduate-level connections. Trigonometry follows, linking the unit circle, radians,
sine/cosine/tangent curves, vector projection, and cosine similarity.
Calculus follows: constants, linear
and power functions, natural logarithms, reciprocals,
exponentials, sine, cosine, and five ML activations (sigmoid, tanh, ReLU, softplus,
and SiLU). Activation comparison overlays all five outputs, derivatives, and
integrals on shared axes. Linear Algebra follows
with a matrix operations and backprop lab.
Vector Calculus and Neural Networks follow as editable JupyterLite notebooks.
Probability Theory and Hypothesis Testing remain planned chapters.

## Mathematical Foundations

The opening chapter contains 71 expandable reminders across arithmetic, number
tables, axioms and proof, algebra, powers and logarithms, geometry, trigonometry,
calculus, linear algebra and optimization, and probability and statistics.
It is a selective refresher, not a complete graduate curriculum. Identities are
distinguished from axioms and theorems, with domain conditions beside the details.

Tables include a multiplication grid with 20 columns and 10 rows, bold headers,
a thick divider after column 10, squares through 25, and all 168 primes
below 1000. Search, topic navigation, and
two canvas diagrams supplement native HTML disclosures. Equations and tables
remain readable without JavaScript. Mobile navigation uses a topic selector.

`chapters/foundations/reference.js` owns the reference content and renders KaTeX
to static MathML at build time; no math library, fonts, Python runtime, or CDN
requests are needed by this chapter. `numbers.js` generates the number tables.
Colocated tests verify those values, rendering, disclosures, search, diagrams,
and desktop/mobile layouts, including JavaScript-disabled reading.

## Build

Requires Node.js 22+, Python 3.9+ for static JupyterLite build/tests, and
Asciidoctor (`gem install asciidoctor`). From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r jupyter/requirements.txt
npm ci
npm test
npm run build
```

The build generates `dist/index.html` from the chapter catalog. Each published
chapter's `content.adoc` becomes `dist/<id>.html`, with a downloadable
`dist/<id>.adoc` source. Interactive chapters have their own JavaScript and CSS
bundles under `dist/chapters/`; common dependency chunks live in `dist/chunks/`.
The contents page does not load Vue, calculus code, or its math/plotting dependencies.
All runtime dependencies are local. No Hugo or server-side runtime is required.
The generated `THIRD-PARTY.txt` retains dependency license notices and should
travel with the deployed directory.

Trigonometry, Calculus, and Linear Algebra execute Python through Pyodide 314.0.6
and NumPy 2.4.6. The first
test/build downloads the NumPy wheel into the installed Pyodide directory;
subsequent runs reuse it. The build verifies wheel SHA-256 checksums against
Pyodide's lockfile and copies the runtime and required wheels into `dist/pyodide/`.
These assets occupy about 18 MB before HTTP compression. They load only when a
Python chapter is opened, not on the contents page. Deploy the entire
directory, including wheels and their embedded license notices; no runtime CDN
is needed for the default lessons. Readers need no local Python installation.

Vector Calculus and Neural Networks use JupyterLite 0.6.4 with the Pyodide kernel 0.6.1 and its
compatible Pyodide 0.27.6 runtime. `jupyter/build.mjs` builds the notebook frontend
and bundles NumPy, Plotly 6.3.1, IPython, and kernel dependencies under
`dist/jupyter/` (about 60 MB before HTTP compression). Runtime wheel checksums
are verified against the Pyodide lockfile. The browser test blocks external
requests while executing and editing the notebook. Additional Python packages
are not fetched automatically from PyPI; bundle them explicitly when adding lessons.

Python, `jupyter-server`, and `ipykernel` in the build environment are only tools
for static contents indexing and local notebook tests. No Python service is
started or deployed. Serve `dist/` over HTTPS (or localhost for development);
JupyterLite uses browser workers and browser storage. Do not open it via `file://`.

## Structure

```text
app.js                     Chapter catalog, book rendering, browser navigation
book.test.mjs              Catalog and navigation tests
shell.html                 Shared document wrapper
style.css                  Book layout, typography, and reference sections
build.mjs                  AsciiDoc pages and chapter-local asset bundles
playwright.config.js       Browser tests against a running static preview
python/
  PythonLab.vue            Workspace, execution status, plots, and value grids
  PythonWorkspace.vue      Reusable editor, output log, and execution controls
  PythonEditor.vue         CodeMirror 6 Python editor with line-number gutters
  ValueGrid.vue            Read-only scalar, vector, and matrix previews
  PythonPlots.vue          Shared plot legends and synchronized navigation
  plots.js                 D3 rendering of Python-produced points only
  plots.css                Responsive plots and legend styling
  runner.py                INPUTS/OUTPUTS/PLOTS validation and bounded output
  runner.test.mjs          Export contract tests using real Pyodide in Node
  runtime-fixture.mjs      Shared Pyodide harness for numerical/export tests
  style.css                Compact scalar, vector, and matrix rendering
  workspace.css            Viewport-height workspace and responsive panels
  runtime.js               Worker requests, execution timeout, cancellation
  worker.js                Local Pyodide + NumPy, cached driver and lesson session
chapters/
  trigonometry/
    content.adoc           Radians, trig identities, projection, and vector products
    index.js               Chapter mount and stylesheet import
    Trigonometry.vue       Linked unit circle, vector diagram, and wave plots
    lesson.py              Float32 trig and vector equations, sampled plot data
    style.css              Responsive chapter diagrams and controls
    python.test.mjs        Geometric identities and singularity checks
    trigonometry.browser.spec.js  Angle units, similarity, and navigation
  calculus/
    content.adoc           Reference notes and Vue mount container
    index.js               Vue mount and chapter stylesheet import
    PythonCalculus.vue     Preset selection and shared PythonLab integration
    expressions.js         NumPy expressions and Python template substitution
    lesson.py.in           Function, derivative, integral sampling template
    composition.py.in      Sum/product/chain-rule sampling template
    comparison.py.in       Activation comparison sampling template
    python.css             Compact example selector
    python.test.mjs        Numerical checks against real Python output
    python.browser.spec.js Editor, plots, navigation, recovery, layouts
  linear-algebra/
    content.adoc           Neural-network shapes, gradients, and reference notes
    index.js               Mount shared PythonLab with default script
    lesson.py              Code-owned arrays, equations, and export dictionaries
    code-first.browser.spec.js  Script-driven values, errors, output, layouts
```

Keep chapter-specific math, UI, state, tests, and assets inside its directory.
Chapters must not import one another. Promote a utility to shared app code only
when multiple chapters actually need the same behavior. There is no global
subject-switching component, client-side router, or shared mutable chapter state.

Interactive chapters use Vue 3 Composition API single-file components with
`<script setup>`. The build uses the official `@vue/compiler-sfc` compiler with
esbuild; templates compile ahead of time, so no runtime template compiler or
server rendering is needed. Keep styles in chapter CSS imported from `index.js`,
not SFC style blocks. Vue and its compiler are pinned to matching versions.

Vue owns controls and displayed values. D3 exclusively owns the SVG descendants
of empty plot containers referenced by Vue. Draw after reactive updates and
disconnect observers and D3 handlers on unmount. Both chapters use local Python,
imported as text by esbuild, for all equations and sampling. Calculus's expression
catalog substitutes only trusted preset expressions or parsed finite numeric
parameters into its Python templates. The editor shows the resulting executable
source, not placeholders. The `.py.in` files are templates, not executable Python
modules; numerical tests execute their generated source. Plot titles and labels
are rendered as text, not HTML.

The editor folds Calculus sampling/export code and Linear Algebra's display-only
exports by default. Show framework reveals the full editable script; folding never
changes execution or the downloadable source. Equations remain visible.

## Adding a chapter

1. Add an entry to the `chapters` array in `app.js`: a unique lowercase hyphenated `id`, `title`,
  `description`, and `status: 'planned'`. Catalog order is book order. The ID
  `index` is reserved for the contents page.
2. Author `chapters/<id>/content.adoc`. Use AsciiDoc sections for the chapter's
  explanations; text-only chapters need no JavaScript.
3. For an interactive chapter, add `index.js`, set `interactive: true` in the
  catalog, and mount a Vue component into a chapter-specific container in the
  AsciiDoc (for example, `<div data-calculus></div>` inside a passthrough block).
  Use `createApp(Component).mount(container)` as in the calculus entry point.
  Import local CSS from the entry point. Esbuild follows local module imports,
  creates a separate chapter bundle, and the build links its emitted CSS.
4. Optionally set `reference` to an AsciiDoc heading ID, without `#`, to enable
  the header's Reference link. Add colocated `*.test.mjs` files for behavior.
5. Set `status: 'published'` when ready, then run `npm test` and `npm run build`.
  The page, contents link, and previous/next published-chapter navigation are
  generated automatically. Beyond the catalog entry, no shared logic or build
  changes are needed.

Use relative links and bundled imports for chapter resources; avoid external
runtime dependencies. Planned entries do not require source files, allowing the
contents to outline chapters such as Hypothesis Testing before authoring begins.

Re-run `npm run build` after edits. The site's Hugo server watches the generated
directory; source files are not compiled automatically by Hugo.

## Browser tests

`npm test` runs real-Pyodide numerical/export tests and book-rendering tests.
For the Vue/D3 integration checks, build and serve the app, then run:

```sh
npx playwright install chromium
npm run test:browser
```

The default preview URL is `http://localhost:1414/app/refresh/`. To test another
static server, set `REFRESH_BASE_URL` to the deployed app's base URL, including
the trailing slash. For example, with the site's default `make serve` port:

```sh
REFRESH_BASE_URL=http://localhost:1313/app/refresh/ npm run test:browser
```

These tests require an already-running preview; they do not start a server.
They cover preset selection, legacy numeric URLs, plot exports, invalid and
numerical integrals, synchronized zoom/pan/reset, navigation, and responsive layouts.
Python checks cover default values and gradients, export validation, changing
labels and dimensions, scalar/vector/matrix previews, and empty groups. Browser
checks also cover Python edits, rejected exports, Stop/timeout recovery, stale
code responses, retry after failed startup, edited-draft protection, and running
with external requests blocked.
Screenshots and failure traces are written under ignored `test-results/`.

## Host or relocate

Copy the contents of `dist/` to any static HTTP host, at any path. Assets and links
are relative, and the app does not use remote fonts, APIs, or CDN dependencies.
Copy this whole source directory (excluding `node_modules/` and `dist/`) to move
the development project. Node and Asciidoctor are build-time requirements only.

The parent website mounts `dist/` at `/app/refresh/`. Direct `hugo` commands
require the app to have been built first; `make build` and `make serve` do this.
Calculus URL fragments select a function, composition rule, or activation comparison.
Legacy finite numeric coefficient/exponent/point/lower parameters initialize the
Python source on page load. Editor drafts are not put into URLs or auto-executed
from them; changing an example immediately replaces the current draft without prompting.
Calculus is now at `calculus.html`; old root URLs containing a `family` parameter
are forwarded there with their hash intact. The catalog's optional
`legacyStateKey` exists for this compatibility path, not for new chapter routing.

## Plot navigation

Trigonometry links a draggable unit circle to Python-computed trig and vector
diagrams. Arrow keys adjust by one degree (Shift: 15); numeric entry supports
degrees/radians. The cross-product diagram shows the signed z component, not
a scalar similarity measure.

Charts use a 16:9 width-to-height ratio, capped at 800px wide.
Major ticks and grid lines adapt to the available space, with finer unlabeled
ticks between them. Calculus presets sample x from -10 to 10 by default;
undefined values, such as logarithms at nonpositive x, remain gaps.

Individual function presets share one plot for the function, tangent, derivative,
and signed integral. The integral is the shaded region between `f(x)` and zero
from `lower` to `point`, with the bounds and Python-computed net value displayed
above it. Teal indicates positive contributions and rose negative contributions;
reversing the bounds reverses those signs. Undefined integrals have no shading.
Shading is limited to the sampled range. Composition and activation comparison
retain their separate synchronized plots.

The Vue chapters share D3 wheel zoom, touch pinch, drag-to-pan, and toolbar navigation.
With any plot focused, `+`/`-` zoom, arrows pan, and `0` resets. Zoom ranges from
0.25x to 32x. Normalized pan offsets survive desktop/mobile layout changes; each
plot retains its own fitted y-scale. Legend checkboxes toggle individual series.

JavaScript only renders the points returned by Python. Zoom does not evaluate
equations or generate additional samples outside the supplied range. Edit the
Python sampling range or sample count and Run to change resolution or coverage.
A manual run resets the view; resize and legend toggles preserve it. Undefined
ordinates break curves rather than connecting across a pole or kink.

Calculus uses a draggable evaluation point on the Function plot, the composition
result h(x), or the first visible activation curve. Drag it horizontally with a
mouse or touch; arrow keys adjust by 0.01 (Shift: 0.1). Precise numeric entry
remains beside the zoom controls. Background dragging still pans the plots.
The handle follows sampled curves, becoming hollow at undefined values. Dragging
is bounded by the samples. Updates retain axes, zoom, and legends; manual Run
refits the axes. Cached curves stay fixed while tangent endpoints, integral
metadata, and numerical readouts update through Python.

## Vector calculus

`chapters/vector-calculus/vector-calculus.ipynb` is the lesson source. Markdown
equations alternate with editable NumPy cells and interactive Plotly figures:
scalar surfaces and contours, directional Taylor approximations, Hessian
eigenvalues, and paired Jacobian input/output diagrams. Bowl, coupled bowl,
saddle, and quartic examples retain analytic derivatives and finite-difference checks.

Drag the surface to rotate it; plots support zoom, pan, data hover, and legend
toggles. These interactions run in JavaScript without rerunning Python. Changing
the function, evaluation point, or direction still requires rerunning the cells.
The notebook renderer handles resizing; Plotly's separate `responsive` mode is
disabled to prevent percentage-height output collapse after zooming in JupyterLite.
Plotly's MIME renderer and all required Python wheels are bundled locally.

The chapter embeds the notebook frontend and links to a full-screen view and a
portable `.ipynb` download. Use Jupyter's Run All to initialize the lesson; edit
the parameter cell and rerun to compare examples. This replaces the custom Vue
editor and drag controls for this chapter only. JupyterLite saves edits in browser
storage, not the repository. Export a notebook to keep a portable copy; browser
copies can take precedence over newer published content.

## Neural networks

`chapters/neural-network/neural-network.ipynb` is an eight-step, 19-cell tutorial for a
5-input, 10-hidden, 4-output classifier with ReLU and softmax. It derives
cross entropy, the softmax Jacobian, the combined `(P - Y) / N` gradient,
affine-layer gradients, the ReLU mask, and bias-corrected Adam updates.
NumPy implements forward and backward passes without autodiff. Training is
float32. Exhaustive checks live in the accompanying test rather than the reading
sequence: float64 finite differences cover all 104 parameters and input gradients,
alongside softmax stability, Adam state, and minibatch averaging checks.

Three Plotly figures cover the dataset projection, clean learning curves, and
noisy-label memorization curves. Forward and backward show matrix operations
directly, without single-operation layer wrappers. The clean experiment uses 320 training and
160 validation examples with training-only standardization. A separate experiment
memorizes 16 balanced but randomly reassigned labels while evaluating against the
original validation labels. Both reset weights and optimizer state. Default runs
reach about 92.5% clean validation accuracy, versus 100% tiny-set training accuracy
and about 13.8% validation accuracy after memorization; numerical results may vary
slightly between runtimes.

Notebook chapters set `notebook: true` in the catalog, keep their canonical
`<id>.ipynb` beside `content.adoc`, and import `jupyter/notebook.js` from their
entry point. The build discovers published notebook chapters and copies them into
the shared JupyterLite contents and downloadable output. The shared notebook shell
fills the viewport below the book header, with full-screen and download icons.

## Matrix operations and backprop

Only the example inputs X and targets Y are hardcoded. The lesson constructs
weight and bias arrays with shapes derived from X and Y, drawn uniformly within
`+/-1 / sqrt(in_features)`, using `np.random.uniform` directly. A single top-level
`np.random.seed(42)` makes initialization and sampled examples reproducible on
Run and Reset. All lesson arrays, predictions, losses, and gradients use float32.

`update(learning_rate)` runs ten explicit SGD steps, computing MSE gradients and
applying `W -= learning_rate * dW` and `b -= learning_rate * db`. Each comparison
starts from the same cached initial weights and ten sampled rows, not the previous
trained result. The default rate is 0.1; X and Y stay fixed. Editing data,
equations, or the seed performs fresh setup on the next run.

Full-data predictions and MSE are evaluated after every update. The four outputs
are Y_before, Y_pred (after update 10), and trained W/b. Only X and Y are displayed
as inputs, in compact, non-stretching matrix previews. Loss progress for steps
0 through 10 appears in the terminal, not in the visualization. The default
run reduces loss substantially; arbitrary data, seeds, or learning rates need
not improve at every step or converge. This demonstrates fitting the training
examples, not generalization. Python owns all computations; the callback returns
the INPUTS/OUTPUTS dictionaries. The framework block at the bottom declares the
live learning-rate parameter and performs the initial update.

Edit arrays, shapes, expressions, and export labels directly in Python. The UI
has no operation selector, editable matrix cells, or hard-coded dependency traces.
There is no numeric input bound such as +/-10000: finite Python values are rendered
with compact formatting, and cell tooltips show the returned numeric value.
Color magnitudes are normalized independently within each displayed preview.

Automatic numerical-gradient checking was removed with the fixed function contract.
Gradients are ordinary output variables, not inferred or graded by the renderer.
Numerical checks can be written explicitly in Python and their results exported
or printed. Softmax and attention remain future lesson content.

## Shared exercise convention

Every Python chapter/exercise supplies a complete runnable default script with
comments describing its inputs, equations, and visualization exports. The shared
contract is two dictionaries, evaluated after the script finishes:

```python
import numpy as np

X = np.array([[1., 2.]], dtype=np.float32)
Y = np.array([[10.]], dtype=np.float32)
np.random.seed(42)
bound = 1 / np.sqrt(X.shape[1])
W = np.random.uniform(-bound, bound, size=(X.shape[1], Y.shape[1])).astype(np.float32)
b = np.random.uniform(-bound, bound, size=(1, Y.shape[1])).astype(np.float32)
Y_pred = X @ W + b
loss = np.mean((Y_pred - Y) ** 2)

# The visualization reads these dictionaries after execution.
# Keys are display labels; values are scalars, vectors, or matrices.
INPUTS = {"X": X, "Y": Y, "W": W, "b": b}
OUTPUTS = {"Y_pred": Y_pred, "loss": loss}
```

- Both `INPUTS` and `OUTPUTS` are required; use `{}` for an empty group.
- Keys are nonempty strings of at most 80 characters. Dictionary order is display
  order; the same label may appear in both groups.
- Values are finite real numeric scalars, NumPy arrays, or rectangular numeric
  lists. Supported dimensions are scalar (0D), vector (1D), and matrix (2D).
- Shapes and dtypes come from Python values, never from chapter-specific JS.
  Vectors display as a row with a `(n,)` label, not as an invented 2D shape.
- Unexported intermediates, modules, and functions are not displayed. No names
  such as X or Y are reserved. Put exports after the computations so reassigned
  variables are represented by their final values. Exports store values, not
  variable-name lookups, following ordinary Python dictionary semantics.
- Each group supports at most 32 entries. Each array preview shows its first
  12 rows/columns, with full shape and a truncation label. This limits rendering,
  not the computation; large previews scroll inside their own area.
- Validation checks the export contract, not mathematical correctness. It does
  not infer equations, dependency graphs, gradients, or application semantics.

New Python chapters can import their `lesson.py` as text and mount the shared
`PythonLab` with `defaultCode` and `title` props, as the Linear Algebra entry does.
The optional `download` prop points to a deployed default script. Keep the lesson
and reference notes chapter-local; reuse `python/runner.py` and the generic UI.

Chapters needing only code and terminal output can use `PythonWorkspace` directly,
without the value grids. It accepts the same `defaultCode`, `title`, and optional
`download` props. It owns its editor, worker, output, cancellation, and resizing.
Its `state` event emits `{ result, busy, stale, error, modified }`; a chapter can use `result`
for its own visualization. The default driver still requires `INPUTS` and `OUTPUTS`
(empty dictionaries are allowed). An optional `driver` prop accepts a Python driver
as text for chapter-specific execution, following the existing worker JSON contract.
The optional `toolbar` slot adds chapter controls without another header bar.
`PythonLab` forwards that slot and event and renders both plot and value exports.

### Low-latency updates

The worker initializes Pyodide, NumPy, and the driver once, and caches the latest
compiled source. Full Run uses a fresh namespace; live controls call a retained
`update` function. Put expensive setup and static plots outside that function:

```python
angle = 45.0

def update(angle):
  theta = np.deg2rad(np.float32(angle))
  return {"INPUTS": {}, "OUTPUTS": {"sin": np.sin(theta)}}

PARAMETERS = {"angle": angle}
globals().update(update(**PARAMETERS))
```

`PARAMETERS` declares up to 32 finite numeric controls passed as keyword arguments.
Return `INPUTS` and `OUTPUTS`; optional `PLOTS` describes the complete collection.
Omitting `PLOTS` retains it. The runner detects sample changes, including in-place
edits, and transfers only changed series/area metadata. Structural changes replace
all plots; caches are bounded by export limits.

`workspace.updateSource(nextSource, { angle: nextAngle })` synchronizes the editor
literal and sends parameters with a session revision. Only the latest pending
update is retained. Other code edits/errors invalidate the session; Stop destroys
it. Calling `updateSource(nextSource)` without parameters requests a full run.
PythonLab supplies numeric controls by default; override its controls/plots slots
for custom UI. Missing, duplicate, or nonliteral assignments disable controls.

### Plot exports

`PLOTS` is optional. Each title maps to named real numeric `(N, 2)` arrays:

```python
PLOTS = {"Function": {"f(x)": np.column_stack((x, f(x)))}}
```

A shaded series can instead supply points and integration metadata:

```python
PLOTS = {"Function": {"f(x)": {
  "points": np.column_stack((x, f(x))),
  "area": {"lower": lower, "upper": point, "value": F(point) - F(lower)},
}}}
```

Area bounds must be finite real scalars; a nonfinite value becomes `null` and
disables shading. Python owns the signed value and validity checks, including
domain continuity. The renderer fills the supplied samples to zero and clips
them to the bounds; it does not perform numerical integration.

Limits are 12 plots, 8 series per plot, 2 to 4097 points per series, and 100000
points overall. Titles and labels must be nonempty strings of at most 80 characters.
X coordinates must be finite; nonfinite Y values serialize as `null` gaps.
Unlike matrix previews, plot arrays are not truncated. INPUTS and OUTPUTS remain
required, but can be empty for plot-only lessons. The renderer never interprets
labels as equations or HTML and does not recompute values in JavaScript.

## Editable Python lessons

The prefilled editor displays the actual Python source used for computation.
It uses locally bundled CodeMirror 6 with Python syntax highlighting, line numbers,
code folding, automatic indentation, undo/redo, search, and responsive line wrapping.
Ctrl/Cmd+Enter runs the code. Tab remains available for leaving the editor.
The code and output terminal appear side by side on desktop, with a default
65% code width. The workspace, including its run toolbar, starts at 50% of the
viewport height, with a 360px minimum. Compact numeric visualizations appear below it. At widths of
800px or less, Code and Output become keyboard-accessible tabs; switching tabs
preserves the draft.
Drag the workspace divider to adjust code width on desktop, or the handle below
the workspace to adjust height on any screen size. Focused handles accept arrow keys
and Home/End; double-click restores the default size. Resizing preserves the draft.
Run Python executes a snapshot of the current draft. The default script populates
the editor and runs once on page load. Further edits wait for Run; they mark the
last successful visualization as stale. Failed runs keep that visualization.
If equation code changes while a run is in progress, its result is discarded and the user
must run the current draft. Reset Python code restores and runs the full default
script. Drafts are not saved across page reloads. The download link provides the
default lesson as a `.py` file. All views are read-only reflections of Python.

Execution uses one module worker per mounted workspace. Startup has a 60-second
timeout; each execution has a 5-second timeout. Stop terminates the worker, and
the next Run initializes a new runtime. No SharedArrayBuffer or special isolation
headers are needed. The output terminal streams labelled stdout and stderr entries in
write order, including partial writes and messages printed before an exception or
Stop. Each stream is bounded to 8,000 characters with a truncation notice; tracebacks
are bounded as well. Output resets at the start of a run and can be cleared manually.
Desktop code and output remain visible together. On mobile, Run stays on the
current tab, errors open Output, and Reset Python code returns to Code. Interactive stdin and terminal
escape-sequence emulation are unsupported: this is a code pad with an output log,
not a full terminal REPL. Startup failures discard the worker so the next Run can
retry initialization; clearing output never marks a failed run's old results current.

Each full run has a fresh lesson namespace, but imports live in a reused interpreter:
this is not full process isolation or a security sandbox. Student code has browser
capabilities through Pyodide's JavaScript bridge. Do not auto-execute code loaded
from URLs, saved snippets, or other users. Untrusted shared-code execution would
require a separate security design. Only the shipped lesson runs automatically
on page load. The worker limits ordinary loops, not all possible memory abuse.

All interactive chapters use the shared Python runner and workspace.
Future chapters supply their own Python source and reference notes, with optional
plot and value exports. Math.js and the JavaScript integration engine are removed.

## Mathematical conventions

Calculus presets expose NumPy expressions for `f`, its analytic derivative `df`,
and an antiderivative `F`. Parameters, bounds, and sampling live in the editor,
not duplicated sliders. Updating `f` does not automatically derive `df` or `F`:
the learner edits those expressions too. Default expressions are checked against
finite differences; arbitrary user edits are not mathematically validated.

- Both Calculus and Linear Algebra use NumPy float32 for computations and sampling.
- `np.log(x)` is the natural logarithm; its real domain is `x > 0`. Trigonometric
  arguments use radians.
- Individual presets shade `f(x)`; activation comparison shows accumulation `A(x) = F(x) - F(lower)`.
  Reversed bounds reverse the total; intervals crossing or touching the reciprocal
  pole are undefined, not assigned a principal value.
- ReLU's derivative at zero is undefined and creates a plot gap. Stable sigmoid,
  softplus and tanh expressions avoid overflow in their saturated tails.
- Softplus and SiLU use `np.trapezoid` with 1025 samples from zero to each argument;
  terminal output marks these as numerical approximations. Other presets use
  analytic primitives evaluated numerically.
- Composition examples expose `f`, `df`, `g`, `dg`, `h`, and `dh`. Addition/product
  contributions and chain-rule factors are printed; chain mode evaluates the
  outer tangent at `g(point)`. Independent y-scales mean tangent angles across
  separate plots are not directly comparable.
- Activation comparison evaluates the same catalog expressions for all five
  functions, derivatives, and integrals, with distinct colors and dash patterns.