import { z } from 'zod';

export const loginSchema = z.object({
  username: z.string().min(3, 'Username must be at least 3 characters'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  date_of_birth: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'Must be YYYY-MM-DD format'),
});

export const preferencesSchema = z.object({
  bankroll: z.number().min(0).optional(),
  max_stake_per_bet: z.number().min(0).optional(),
  risk_tolerance: z.enum(['LOW', 'MEDIUM', 'HIGH']).optional(),
  confidence_threshold: z.number().min(0).max(1).optional(),
  kelly_fraction: z.number().min(0).max(1).optional(),
  favorite_sports: z.array(z.string()).optional(),
  notification_enabled: z.boolean().optional(),
});

export const accumulatorSchema = z.object({
  min_legs: z.number().int().min(2).max(10),
  max_legs: z.number().int().min(2).max(10),
  min_total_odds: z.number().min(1.5),
  max_total_odds: z.number().min(2.0),
  confidence_threshold: z.number().min(0.5).max(0.95),
}).refine((data) => data.max_legs >= data.min_legs, {
  message: 'Max legs must be >= min legs',
  path: ['max_legs'],
});

export const betRecordSchema = z.object({
  bookmaker: z.string().min(1, 'Bookmaker is required'),
  bet_type: z.string().min(1, 'Bet type is required'),
  selection: z.string().min(1, 'Selection is required'),
  odds: z.number().gt(1, 'Odds must be greater than 1'),
  stake: z.number().gt(0, 'Stake must be greater than 0'),
  notes: z.string().optional(),
});

export type LoginFormData = z.infer<typeof loginSchema>;
export type PreferencesFormData = z.infer<typeof preferencesSchema>;
export type AccumulatorFormData = z.infer<typeof accumulatorSchema>;
export type BetRecordFormData = z.infer<typeof betRecordSchema>;
