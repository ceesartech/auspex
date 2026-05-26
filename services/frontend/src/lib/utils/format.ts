import { format, formatDistanceToNow, parseISO } from 'date-fns';

export function formatCurrency(amount: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
  }).format(amount);
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatOdds(odds: number, format: 'decimal' | 'american' | 'fractional' = 'decimal'): string {
  if (format === 'decimal') return odds.toFixed(2);
  if (format === 'american') {
    if (odds >= 2) return `+${Math.round((odds - 1) * 100)}`;
    return `-${Math.round(100 / (odds - 1))}`;
  }
  // Fractional
  const numerator = Math.round((odds - 1) * 100);
  const denominator = 100;
  const gcd = getGCD(numerator, denominator);
  return `${numerator / gcd}/${denominator / gcd}`;
}

function getGCD(a: number, b: number): number {
  return b === 0 ? a : getGCD(b, a % b);
}

export function formatDateTime(dateString: string): string {
  try {
    const date = parseISO(dateString);
    return format(date, 'MMM d, yyyy HH:mm');
  } catch {
    return dateString;
  }
}

export function formatDate(dateString: string): string {
  try {
    const date = parseISO(dateString);
    return format(date, 'MMM d, yyyy');
  } catch {
    return dateString;
  }
}

export function formatRelativeTime(dateString: string): string {
  try {
    const date = parseISO(dateString);
    return formatDistanceToNow(date, { addSuffix: true });
  } catch {
    return dateString;
  }
}

export function formatNumber(num: number, decimals = 0): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(num);
}

export function formatConfidenceLevel(level: number | string): string {
  if (typeof level === 'number') {
    if (level >= 0.7) return 'HIGH';
    if (level >= 0.5) return 'MEDIUM';
    return 'LOW';
  }
  return level.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function getOutcomeColor(outcome: string): string {
  const lower = outcome.toLowerCase();
  if (lower.includes('home') || lower === 'won') {
    return 'text-green-600 dark:text-green-400';
  }
  if (lower.includes('away') || lower === 'lost') {
    return 'text-red-600 dark:text-red-400';
  }
  if (lower.includes('draw') || lower === 'pending') {
    return 'text-yellow-600 dark:text-yellow-400';
  }
  return 'text-muted-foreground';
}
