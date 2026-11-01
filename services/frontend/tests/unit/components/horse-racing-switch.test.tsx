import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { Sidebar } from '@/components/layout/sidebar';
import { Header } from '@/components/layout/header';
import RacesPage from '@/app/(authenticated)/races/page';
import RaceDetailPage from '@/app/(authenticated)/races/[id]/page';
import { useAuthStore } from '@/lib/store/auth-store';

// The navigation and the /races pages under HORSE_RACING_ENABLED
// (@/lib/sport-switches), both ways. `mockHorseRacing` undefined keeps the
// committed value; true / false flips the one constant for a test. The race
// hooks are mocks: a paused page must not call them at all.

let mockHorseRacing: boolean | undefined;
jest.mock('@/lib/sport-switches', () => {
  const actual = jest.requireActual('@/lib/sport-switches');
  const value = () => (mockHorseRacing === undefined ? actual.HORSE_RACING_ENABLED : mockHorseRacing);
  return {
    get HORSE_RACING_ENABLED() {
      return value();
    },
    isRouteEnabled: (href: string) => actual.isRouteEnabled(href, value()),
  };
});

jest.mock('next/navigation', () => ({
  usePathname: () => '/',
  useRouter: () => ({ push: jest.fn(), replace: jest.fn() }),
}));

jest.mock('@/lib/hooks/use-auth', () => ({
  useLogout: () => ({ mutate: jest.fn() }),
}));

const mockUseRaces = jest.fn();
const mockUseRaceDetail = jest.fn();
jest.mock('@/lib/hooks/use-races', () => ({
  useRaces: (...args: unknown[]) => mockUseRaces(...args),
  useRaceDetail: (...args: unknown[]) => mockUseRaceDetail(...args),
}));

function renderNavigation() {
  useAuthStore.setState({ isAuthenticated: true, token: 't', user: null });
  render(
    <>
      <Sidebar />
      <Header />
    </>
  );
}

beforeEach(() => {
  mockHorseRacing = undefined;
  mockUseRaces.mockReset();
  mockUseRaceDetail.mockReset();
});

describe('horse racing dormant (the committed switch)', () => {
  it('is what the committed constant says', () => {
    expect(jest.requireActual('@/lib/sport-switches').HORSE_RACING_ENABLED).toBe(false);
  });

  it('the sidebar and the header offer no Race cards entry', () => {
    renderNavigation();
    expect(screen.queryByRole('link', { name: /race cards/i })).not.toBeInTheDocument();
    expect(document.querySelector('a[href="/races"]')).toBeNull();
    expect(screen.getAllByRole('link', { name: 'Predictions' })).toHaveLength(2);
  });

  it('/races shows the paused notice and fetches nothing', () => {
    render(<RacesPage />);
    expect(screen.getByTestId('races-paused')).toBeInTheDocument();
    expect(screen.getByText('Horse racing is paused')).toBeInTheDocument();
    expect(screen.queryByText(/no upcoming races/i)).not.toBeInTheDocument();
    expect(mockUseRaces).not.toHaveBeenCalled();
  });

  it('a /races/[id] deep link shows the same notice and fetches nothing', () => {
    render(<RaceDetailPage params={{ id: 'race-1' }} />);
    expect(screen.getByText('Horse racing is paused')).toBeInTheDocument();
    expect(mockUseRaceDetail).not.toHaveBeenCalled();
  });
});

describe('horse racing turned back on (the one constant flipped)', () => {
  beforeEach(() => {
    mockHorseRacing = true;
  });

  it('the Race cards entry returns to the sidebar and the header', () => {
    renderNavigation();
    expect(screen.getAllByRole('link', { name: /race cards/i })).toHaveLength(2);
  });

  it('/races fetches and lists race cards again', () => {
    mockUseRaces.mockReturnValue({ data: [], isLoading: false, error: null, refetch: jest.fn() });
    render(<RacesPage />);
    expect(mockUseRaces).toHaveBeenCalledWith({ status: 'scheduled', days_ahead: 2, limit: 200 });
    expect(screen.getByText('No upcoming races')).toBeInTheDocument();
    expect(screen.queryByTestId('races-paused')).not.toBeInTheDocument();
  });

  it('/races/[id] fetches the race again', () => {
    mockUseRaceDetail.mockReturnValue({ data: undefined, isLoading: true, error: null, refetch: jest.fn() });
    render(<RaceDetailPage params={{ id: 'race-1' }} />);
    expect(mockUseRaceDetail).toHaveBeenCalledWith('race-1');
    expect(screen.queryByTestId('races-paused')).not.toBeInTheDocument();
  });
});
