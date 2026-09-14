export const multiplicationGrid = Array.from({ length: 10 }, (_, rowIndex) =>
  Array.from({ length: 20 }, (_, columnIndex) => (rowIndex + 1) * (columnIndex + 1)));

export const powers = Array.from({ length: 25 }, (_, index) => {
  const base = index + 1;
  return { base, square: base ** 2 };
});

export function primesUpTo(limit) {
  const composite = new Uint8Array(limit + 1);
  const primes = [];
  for (let candidate = 2; candidate <= limit; candidate++) {
    if (composite[candidate]) continue;
    primes.push(candidate);
    for (let multiple = candidate * candidate; multiple <= limit; multiple += candidate) composite[multiple] = 1;
  }
  return primes;
}

export const primes = primesUpTo(1000);