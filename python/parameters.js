import { pythonLanguage } from '@codemirror/lang-python';

export function numericParameters(source, names) {
  const parameters = {};
  for (const node of pythonLanguage.parser.parse(source).topNode.getChildren('AssignStatement')) {
    const name = source.slice(node.firstChild.from, node.firstChild.to);
    if (!names.includes(name) || node.firstChild.name !== 'VariableName') continue;
    const operator = node.firstChild.nextSibling;
    const value = operator?.nextSibling;
    const number = value && Number(source.slice(value.from, value.to).replaceAll('_', ''));
    parameters[name] = Object.hasOwn(parameters, name) || operator?.name !== 'AssignOp' || !value || value.nextSibling || !['Number', 'UnaryExpression'].includes(value.name) || !Number.isFinite(number)
      ? null : { from: value.from, to: value.to, value: number };
  }
  return parameters;
}