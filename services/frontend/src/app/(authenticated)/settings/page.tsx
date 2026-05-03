'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { useSettingsStore } from '@/lib/store/settings-store';
import { useUserPreferences, useUpdatePreferences } from '@/lib/hooks/use-auth';
import { preferencesSchema, type PreferencesFormData } from '@/lib/utils/validation';
import { toast } from '@/components/ui/toast';
import { LoadingSpinner } from '@/components/shared/loading';

export default function SettingsPage() {
  const { theme, setTheme, oddsFormat, setOddsFormat, currency, setCurrency } = useSettingsStore();
  const { data: preferences, isLoading } = useUserPreferences();
  const updatePreferences = useUpdatePreferences();

  const { register, handleSubmit, formState: { errors } } = useForm<PreferencesFormData>({
    resolver: zodResolver(preferencesSchema),
    values: preferences ? {
      bankroll: (preferences as any).bankroll?.value ?? 1000,
      max_stake_per_bet: (preferences as any).max_stake_per_bet?.value ?? 50,
      risk_tolerance: (preferences as any).risk_tolerance?.value ?? 'MEDIUM',
      confidence_threshold: (preferences as any).confidence_threshold?.value ?? 0.6,
      kelly_fraction: (preferences as any).kelly_fraction?.value ?? 0.25,
    } : undefined,
  });

  const onSubmit = async (data: PreferencesFormData) => {
    try {
      await updatePreferences.mutateAsync(data);
      toast('Preferences updated successfully!');
    } catch {
      toast('Failed to update preferences');
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">Manage your preferences</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Display Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <label className="text-sm font-medium">Theme</label>
              <Select
                options={[
                  { value: 'light', label: 'Light' },
                  { value: 'dark', label: 'Dark' },
                  { value: 'system', label: 'System' },
                ]}
                value={theme}
                onChange={(e) => setTheme(e.target.value as any)}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Odds Format</label>
              <Select
                options={[
                  { value: 'decimal', label: 'Decimal (1.50)' },
                  { value: 'american', label: 'American (-200)' },
                  { value: 'fractional', label: 'Fractional (1/2)' },
                ]}
                value={oddsFormat}
                onChange={(e) => setOddsFormat(e.target.value as any)}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Currency</label>
              <Select
                options={[
                  { value: 'USD', label: 'USD ($)' },
                  { value: 'EUR', label: 'EUR' },
                  { value: 'GBP', label: 'GBP' },
                ]}
                value={currency}
                onChange={(e) => setCurrency(e.target.value)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <form onSubmit={handleSubmit(onSubmit)}>
          <CardHeader>
            <CardTitle className="text-lg">Betting Preferences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {isLoading ? (
              <div className="flex justify-center py-8">
                <LoadingSpinner />
              </div>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">Bankroll ($)</label>
                  <Input
                    type="number"
                    {...register('bankroll', { valueAsNumber: true })}
                    error={errors.bankroll?.message}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Max Stake Per Bet ($)</label>
                  <Input
                    type="number"
                    {...register('max_stake_per_bet', { valueAsNumber: true })}
                    error={errors.max_stake_per_bet?.message}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Risk Tolerance</label>
                  <Select
                    options={[
                      { value: 'LOW', label: 'Low' },
                      { value: 'MEDIUM', label: 'Medium' },
                      { value: 'HIGH', label: 'High' },
                    ]}
                    {...register('risk_tolerance')}
                    error={errors.risk_tolerance?.message}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Confidence Threshold</label>
                  <Input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    {...register('confidence_threshold', { valueAsNumber: true })}
                    error={errors.confidence_threshold?.message}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Kelly Fraction</label>
                  <Input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    {...register('kelly_fraction', { valueAsNumber: true })}
                    error={errors.kelly_fraction?.message}
                  />
                </div>
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button type="submit" loading={updatePreferences.isPending}>
              Save Preferences
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
