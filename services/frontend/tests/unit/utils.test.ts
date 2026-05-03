import {
  formatCurrency,
  formatPercentage,
  formatOdds,
  formatNumber,
  formatConfidenceLevel,
  getOutcomeColor,
} from '@/lib/utils/format';
import {
  calculateKellyCriterion,
  calculateExpectedValue,
  calculateImpliedProbability,
  calculateAccumulatorOdds,
  calculatePotentialReturn,
  calculateROI,
  isValueBet,
} from '@/lib/utils/calculations';

describe('Format Utils', () => {
  describe('formatCurrency', () => {
    it('formats USD correctly', () => {
      expect(formatCurrency(1234.56)).toBe('$1,234.56');
    });

    it('formats negative amounts', () => {
      expect(formatCurrency(-50.00)).toBe('-$50.00');
    });

    it('formats zero', () => {
      expect(formatCurrency(0)).toBe('$0.00');
    });
  });

  describe('formatPercentage', () => {
    it('formats decimal as percentage', () => {
      expect(formatPercentage(0.756)).toBe('75.6%');
    });

    it('formats zero', () => {
      expect(formatPercentage(0)).toBe('0.0%');
    });

    it('formats 100%', () => {
      expect(formatPercentage(1)).toBe('100.0%');
    });
  });

  describe('formatOdds', () => {
    it('formats decimal odds', () => {
      expect(formatOdds(2.50, 'decimal')).toBe('2.50');
    });

    it('formats american odds for favorites', () => {
      expect(formatOdds(1.50, 'american')).toBe('-200');
    });

    it('formats american odds for underdogs', () => {
      expect(formatOdds(3.00, 'american')).toBe('+200');
    });

    it('formats fractional odds', () => {
      expect(formatOdds(2.50, 'fractional')).toBe('3/2');
    });
  });

  describe('formatConfidenceLevel', () => {
    it('returns HIGH for >= 0.7', () => {
      expect(formatConfidenceLevel(0.8)).toBe('HIGH');
    });

    it('returns MEDIUM for >= 0.5', () => {
      expect(formatConfidenceLevel(0.6)).toBe('MEDIUM');
    });

    it('returns LOW for < 0.5', () => {
      expect(formatConfidenceLevel(0.3)).toBe('LOW');
    });
  });

  describe('getOutcomeColor', () => {
    it('returns green for home win', () => {
      expect(getOutcomeColor('Home Win')).toContain('green');
    });

    it('returns yellow for draw', () => {
      expect(getOutcomeColor('Draw')).toContain('yellow');
    });

    it('returns red for away win', () => {
      expect(getOutcomeColor('Away Win')).toContain('red');
    });
  });
});

describe('Calculation Utils', () => {
  describe('calculateKellyCriterion', () => {
    it('calculates kelly correctly', () => {
      const result = calculateKellyCriterion(0.6, 2.5);
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    });

    it('returns 0 for negative edge', () => {
      const result = calculateKellyCriterion(0.1, 1.5);
      expect(result).toBe(0);
    });
  });

  describe('calculateExpectedValue', () => {
    it('calculates positive EV', () => {
      const ev = calculateExpectedValue(0.6, 2.5);
      expect(ev).toBeGreaterThan(0);
    });

    it('calculates negative EV', () => {
      const ev = calculateExpectedValue(0.3, 2.0);
      expect(ev).toBeLessThan(0);
    });
  });

  describe('calculateImpliedProbability', () => {
    it('converts decimal odds to probability', () => {
      expect(calculateImpliedProbability(2.0)).toBeCloseTo(0.5);
    });

    it('converts 1.5 odds correctly', () => {
      expect(calculateImpliedProbability(1.5)).toBeCloseTo(0.667, 2);
    });
  });

  describe('calculateAccumulatorOdds', () => {
    it('multiplies odds correctly', () => {
      expect(calculateAccumulatorOdds([2.0, 3.0])).toBe(6.0);
    });

    it('handles single leg', () => {
      expect(calculateAccumulatorOdds([2.5])).toBe(2.5);
    });

    it('handles empty array', () => {
      expect(calculateAccumulatorOdds([])).toBe(1);
    });
  });

  describe('calculatePotentialReturn', () => {
    it('calculates return correctly', () => {
      expect(calculatePotentialReturn(10, 2.5)).toBe(25);
    });
  });

  describe('calculateROI', () => {
    it('calculates positive ROI', () => {
      // profit=50, totalStaked=100 → 50%
      expect(calculateROI(50, 100)).toBeCloseTo(0.5);
    });

    it('calculates negative ROI', () => {
      // profit=-20, totalStaked=100 → -20%
      expect(calculateROI(-20, 100)).toBeCloseTo(-0.2);
    });

    it('returns 0 for zero staked', () => {
      expect(calculateROI(100, 0)).toBe(0);
    });
  });

  describe('isValueBet', () => {
    it('returns true for value bet', () => {
      expect(isValueBet(0.6, 2.5)).toBe(true);
    });

    it('returns false for non-value bet', () => {
      expect(isValueBet(0.3, 2.0)).toBe(false);
    });
  });
});
