export function calculateKellyCriterion(
  probability: number,
  odds: number,
  fraction: number = 0.25
): number {
  const q = 1 - probability;
  const b = odds - 1;
  const kelly = (probability * b - q) / b;
  return Math.max(0, kelly * fraction);
}

export function calculateExpectedValue(probability: number, odds: number): number {
  return probability * odds - 1;
}

export function calculateImpliedProbability(odds: number): number {
  return 1 / odds;
}

export function calculateAccumulatorOdds(individualOdds: number[]): number {
  return individualOdds.reduce((total, odds) => total * odds, 1);
}

export function calculatePotentialReturn(stake: number, odds: number): number {
  return stake * odds;
}

export function calculateProfit(stake: number, returns: number): number {
  return returns - stake;
}

export function calculateROI(profit: number, totalStaked: number): number {
  if (totalStaked === 0) return 0;
  return profit / totalStaked;
}

export function isValueBet(probability: number, odds: number): boolean {
  return calculateExpectedValue(probability, odds) > 0;
}
