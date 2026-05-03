'use client';

import { AccumulatorBuilder } from '@/components/accumulator/accumulator-builder';

export default function AccumulatorPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Accumulator Builder</h1>
        <p className="text-muted-foreground">
          Build optimized accumulators using AI recommendations
        </p>
      </div>
      <AccumulatorBuilder />
    </div>
  );
}
