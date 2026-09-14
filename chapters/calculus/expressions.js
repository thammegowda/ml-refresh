const sigmoidHelper = 'def sigmoid(x):\n    return np.exp(-np.logaddexp(0, -x))\n';

export const families = {
  constant: { name: 'Constant', value: 'np.full_like(x, coefficient)', derivative: 'np.zeros_like(x)', integral: 'coefficient * x' },
  linear: { name: 'Linear', value: 'coefficient * x', derivative: 'np.full_like(x, coefficient)', integral: 'coefficient * x ** 2 / 2' },
  power: { name: 'Power', value: 'coefficient * x ** exponent', derivative: 'coefficient * exponent * x ** (exponent - 1)', integral: 'coefficient * x ** (exponent + 1) / (exponent + 1)' },
  log: { name: 'Natural logarithm', value: 'np.where(x > 0, np.log(x), np.nan)', derivative: 'np.where(x > 0, 1 / x, np.nan)', integral: 'np.where(x > 0, x * np.log(x) - x, np.nan)', connected: '(lower > 0) & (x > 0)', lower: 1, point: 2 },
  reciprocal: { name: 'Reciprocal', value: 'np.where(x != 0, 1 / x, np.nan)', derivative: 'np.where(x != 0, -1 / x ** 2, np.nan)', integral: 'np.log(np.abs(x))', connected: 'lower * x > 0', lower: 1, point: 2 },
  exp: { name: 'Exponential', value: 'np.exp(x)', derivative: 'np.exp(x)', integral: 'np.exp(x)' },
  sin: { name: 'Sine', value: 'np.sin(x)', derivative: 'np.cos(x)', integral: '-np.cos(x)' },
  cos: { name: 'Cosine', value: 'np.cos(x)', derivative: '-np.sin(x)', integral: 'np.sin(x)' },
  sigmoid: { name: 'Sigmoid', activation: true, value: 'sigmoid(x)', derivative: 'sigmoid(x) * sigmoid(-x)', integral: 'np.logaddexp(0, x)' },
  tanh: { name: 'Tanh', activation: true, value: 'np.tanh(x)', derivative: '1 - np.tanh(x) ** 2', integral: 'np.logaddexp(x, -x) - np.log(np.float32(2))' },
  relu: { name: 'ReLU', activation: true, value: 'np.maximum(0, x)', derivative: 'np.where(x == 0, np.nan, np.where(x < 0, np.zeros_like(x), np.ones_like(x)))', integral: 'np.maximum(0, x) ** 2 / 2' },
  softplus: { name: 'Softplus', activation: true, value: 'np.logaddexp(0, x)', derivative: 'sigmoid(x)', integral: 'np.trapezoid(f(np.linspace(0, x, 1025, dtype=np.float32)), np.linspace(0, x, 1025, dtype=np.float32), axis=0)', numerical: true },
  silu: { name: 'SiLU / Swish', activation: true, value: 'x * sigmoid(x)', derivative: 'sigmoid(x) + x * sigmoid(x) * sigmoid(-x)', integral: 'np.trapezoid(f(np.linspace(0, x, 1025, dtype=np.float32)), np.linspace(0, x, 1025, dtype=np.float32), axis=0)', numerical: true },
};

export const rules = {
  sum: { name: 'Addition', value: 'f(x) + g(x)', derivative: 'df(x) + dg(x)' },
  product: { name: 'Product', value: 'f(x) * g(x)', derivative: 'df(x) * g(x) + f(x) * dg(x)' },
  chain: { name: 'Chain', value: 'f(g(x))', derivative: 'df(g(x)) * dg(x)' },
};

export function functionCode(template, key, parameters = {}) {
  const family = families[key];
  if (!family) throw new Error('Unknown function family');
  return substitute(template, {
    HELPERS: [family.value, family.derivative, family.integral].some(expression => expression.includes('sigmoid(')) ? sigmoidHelper : '',
    VALUE: family.value, DERIVATIVE: family.derivative, INTEGRAL: family.integral,
    CONNECTED: family.connected ?? 'np.isfinite(x)',
    COEFFICIENT: parameters.coefficient ?? 1, EXPONENT: parameters.exponent ?? 2,
    LOWER: parameters.lower ?? family.lower ?? 0,
    POINT: parameters.point ?? family.point ?? 1,
    NUMERICAL: family.numerical ? 'True' : 'False',
  });
}

export function compositionCode(template, rule, outer = 'sin', inner = 'power') {
  if (!rules[rule] || !families[outer] || !families[inner]) throw new Error('Unknown composition preset');
  return substitute(template, {
    HELPERS: [families[outer].value, families[outer].derivative, families[inner].value, families[inner].derivative].some(expression => expression.includes('sigmoid(')) ? sigmoidHelper : '',
    OUTER: families[outer].value, OUTER_DERIVATIVE: families[outer].derivative,
    INNER: families[inner].value, INNER_DERIVATIVE: families[inner].derivative,
    VALUE: rules[rule].value, DERIVATIVE: rules[rule].derivative,
    OUTER_INPUT: rule === 'chain' ? 'g(point)' : 'point',
    FACTORS: rule === 'chain' ? 'df(g(point)), dg(point)' : rule === 'product' ? 'df(point) * g(point), f(point) * dg(point)' : 'df(point), dg(point)',
  });
}

export function comparisonCode(template) {
  return substitute(template, {
    HELPERS: sigmoidHelper,
    FUNCTIONS: Object.values(families).filter(family => family.activation).map(family =>
      `    ${JSON.stringify(family.name)}: (lambda x: ${family.value}, lambda x: ${family.derivative}, lambda x: ${family.integral}),`).join('\n'),
  });
}

function substitute(template, expressions) {
  return template.replace(/__([A-Z_]+)__/g, (_, key) => {
    if (!Object.hasOwn(expressions, key)) throw new Error(`Missing Python expression: ${key}`);
    return String(expressions[key]);
  });
}