import test from 'node:test';
import assert from 'node:assert/strict';
import { multiplicationGrid, powers, primes } from './numbers.js';
import { sections, renderReference } from './reference.js';
import { chapters } from '../../app.js';

test('reference tables contain every requested integer and correct result', () => {
  assert.equal(multiplicationGrid.length, 10);
  for (const [rowIndex, products] of multiplicationGrid.entries()) {
    assert.equal(products.length, 20);
    for (const [columnIndex, product] of products.entries()) {
      assert.equal(product, (rowIndex + 1) * (columnIndex + 1));
    }
  }
  assert.equal(powers.length, 25);
  for (const [index, entry] of powers.entries()) {
    assert.equal(entry.base, index + 1);
    assert.equal(entry.square, entry.base ** 2);
  }
  assert.equal(primes.length, 168);
  assert.equal(primes[0], 2);
  assert.equal(primes.at(-1), 997);
  for (let value = 2; value <= 1000; value++) {
    const isPrime = !Array.from({ length: Math.max(0, Math.floor(Math.sqrt(value)) - 1) }, (_, index) => index + 2).some(divisor => value % divisor === 0);
    assert.equal(primes.includes(value), isPrime, `Primality of ${value}`);
  }
});

test('reference renders unique disclosures and preserves LaTeX notation', () => {
  assert.equal(chapters[0].id, 'foundations');
  const items = sections.flatMap(section => section.items);
  const ids = [...sections, ...items].map(item => item.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.equal(items.length, 71);
  assert.ok(!items.some(item => item.id === 'cubes'));
  const html = renderReference();
  assert.equal((html.match(/<details /g) ?? []).length, items.length);
  assert.equal((html.match(/<summary>/g) ?? []).length, items.length);
  const annotations = [...html.matchAll(/<annotation encoding="application\/x-tex">(.*?)<\/annotation>/gs)].map(match => match[1]);
  assert.ok(annotations.length > 300);
  for (const formula of annotations) assert.doesNotMatch(formula, /[\x00-\x1f]/, `Unexpected escaped character in ${formula}`);
  assert.ok(annotations.includes(String.raw`c\ne0`));
  assert.ok(annotations.includes(String.raw`\pi r^2`));
  const multiplication = items.find(item => item.id === 'multiplication').detail;
  assert.equal((multiplication.match(/<td>/g) ?? []).length, 200);
  assert.equal((multiplication.match(/scope="row"/g) ?? []).length, 10);
  assert.equal((multiplication.match(/scope="col"/g) ?? []).length, 20);
  for (const id of ['square-sum', 'square-difference', 'cube-sum', 'cube-difference', 'difference-squares', 'binomial', 'derivatives', 'integrals', 'euclid', 'special-angles']) {
    assert.ok(items.some(item => item.id === id));
  }
  assert.match(html, /nonzero|nonnegative/);
});

test('featured polynomial identities hold for positive, negative, and zero inputs', () => {
  for (const first of [-5, -1, 0, 2, 7]) {
    for (const second of [-3, 0, 1, 4]) {
      assert.equal((first + second) ** 2, first ** 2 + 2 * first * second + second ** 2);
      assert.equal((first - second) ** 2, first ** 2 - 2 * first * second + second ** 2);
      assert.equal((first + second) ** 3, first ** 3 + 3 * first ** 2 * second + 3 * first * second ** 2 + second ** 3);
      assert.equal((first - second) ** 3, first ** 3 - 3 * first ** 2 * second + 3 * first * second ** 2 - second ** 3);
      assert.ok(first ** 2 - second ** 2 === (first + second) * (first - second));
    }
  }
});