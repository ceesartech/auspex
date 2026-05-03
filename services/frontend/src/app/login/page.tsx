'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { loginSchema, type LoginFormData } from '@/lib/utils/validation';
import { useLogin } from '@/lib/hooks/use-auth';
import { toast } from '@/components/ui/toast';

export default function LoginPage() {
  const login = useLogin();
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormData) => {
    try {
      await login.mutateAsync(data);
    } catch {
      toast('Login failed. Please check your credentials.');
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">CeesarBet Predict</CardTitle>
          <CardDescription>Sign in to access your betting analytics</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div>
              <label className="text-sm font-medium">Username</label>
              <Input
                {...register('username')}
                placeholder="Enter your username"
                error={errors.username?.message}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Password</label>
              <Input
                type="password"
                {...register('password')}
                placeholder="Enter your password"
                error={errors.password?.message}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Date of Birth</label>
              <Input
                type="date"
                {...register('date_of_birth')}
                error={errors.date_of_birth?.message}
              />
            </div>
            <Button type="submit" className="w-full" loading={login.isPending}>
              Sign In
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
