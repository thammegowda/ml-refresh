// Shared KaTeX macros; Appendix A documents the notation. Symbol macros expand to a group so
// they also work as bare subscripts, e.g. p_\vtheta.
const lower = 'abcdefghijklmnopqrstuvwxyz'.split('');
const greek = ['alpha', 'beta', 'delta', 'epsilon', 'eta', 'lambda', 'mu', 'pi', 'sigma', 'theta', 'phi'];

export const macros = Object.freeze({
  '\\R': '{\\mathbb{R}}',
  '\\E': '{\\mathbb{E}}',
  '\\Var': '\\operatorname{Var}',
  '\\Cov': '\\operatorname{Cov}',
  '\\KL': 'D_{\\mathrm{KL}}',
  '\\softmax': '\\operatorname{softmax}',
  '\\logsumexp': '\\operatorname{logsumexp}',
  '\\diag': '\\operatorname{diag}',
  '\\sign': '\\operatorname{sign}',
  '\\argmax': '\\operatorname*{arg\\,max}',
  '\\argmin': '\\operatorname*{arg\\,min}',
  '\\T': '{\\top}',
  '\\one': '{\\mathbf{1}}',
  '\\dd': '\\mathrm{d}',
  '\\mSigma': '{\\boldsymbol{\\Sigma}}',
  ...Object.fromEntries(lower.map((letter) => [`\\v${letter}`, `{\\mathbf{${letter}}}`])),
  ...Object.fromEntries(lower.map((letter) => [`\\m${letter.toUpperCase()}`, `{\\mathbf{${letter.toUpperCase()}}}`])),
  ...Object.fromEntries(greek.map((name) => [`\\v${name}`, `{\\boldsymbol{\\${name}}}`])),
});
