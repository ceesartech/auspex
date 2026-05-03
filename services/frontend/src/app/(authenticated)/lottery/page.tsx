'use client';

import { useState } from 'react';
import { useLotteryDraws, useLotteryAnalysis } from '@/lib/hooks/use-recommendations';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { EmptyState } from '@/components/shared/empty-state';
import { formatCurrency, formatDate } from '@/lib/utils/format';
import { Ticket } from 'lucide-react';

export default function LotteryPage() {
  const [game, setGame] = useState('powerball');
  const { data: draws, isLoading, error, refetch } = useLotteryDraws(game, 20);
  const { data: analysis } = useLotteryAnalysis(game);

  if (isLoading) return <LoadingPage message="Loading lottery data..." />;
  if (error) return <ErrorDisplay message="Failed to load lottery data" onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Lottery</h1>
          <p className="text-muted-foreground">Draw history and number analysis</p>
        </div>
        <Select
          options={[
            { value: 'powerball', label: 'Powerball' },
            { value: 'mega_millions', label: 'Mega Millions' },
          ]}
          value={game}
          onChange={(e) => setGame(e.target.value)}
          className="w-[180px]"
        />
      </div>

      {analysis && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-green-600">Hot Numbers</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {analysis.hot_numbers?.slice(0, 10).map((n) => (
                  <Badge key={n.number} variant="success">
                    {n.number} ({n.frequency})
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-blue-600">Cold Numbers</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {analysis.cold_numbers?.slice(0, 10).map((n) => (
                  <Badge key={n.number} variant="secondary">
                    {n.number} ({n.frequency})
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-orange-600">Overdue Numbers</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {analysis.overdue_numbers?.slice(0, 10).map((n) => (
                  <Badge key={n.number} variant="warning">
                    {n.number} ({n.draws_since_last}d)
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recent Draws</CardTitle>
        </CardHeader>
        <CardContent>
          {!draws?.length ? (
            <EmptyState
              icon={<Ticket className="h-8 w-8" />}
              title="No draws available"
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>Numbers</TableHead>
                  <TableHead>Bonus</TableHead>
                  <TableHead>Multiplier</TableHead>
                  <TableHead>Jackpot</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {draws.map((draw) => (
                  <TableRow key={draw.draw_id}>
                    <TableCell>{formatDate(draw.draw_date)}</TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        {draw.numbers?.map((n: number) => (
                          <span key={n} className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                            {n}
                          </span>
                        ))}
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-destructive text-xs font-bold text-destructive-foreground">
                        {draw.bonus_number}
                      </span>
                    </TableCell>
                    <TableCell>{draw.multiplier || '-'}</TableCell>
                    <TableCell>
                      {draw.jackpot_amount ? formatCurrency(draw.jackpot_amount) : '-'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
