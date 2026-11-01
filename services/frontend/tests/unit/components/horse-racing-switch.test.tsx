import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { Sidebar } from '@/components/layout/sidebar';
import { Header } from '@/components/layout/header';
import { PredictionFilters } from '@/components/predictions/prediction-filters';
import { RecommendationFilters } from '@/components/recommendations/recommendation-filters';
import RacesPage from '@/app/(authenticated)/races/page';
import RaceDetailPage from '@/app/(authenticated)/races/[id]/page';
import { useAuthStore } from '@/lib/store/auth-store';

// Everything HORSE_RACING_ENABLED (@/lib/sport-switches) controls, both ways:
// the navigation, the /races pages and the two filters that offer horse
// racing. `mockHorseRacing` undefined keeps the committed value; true / false
// forces the one constant for a test. No test pins the committed value, so
// flipping it (with the DAG switch it mirrors) needs no test edit. The race
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
    isSportEnabled: (sport: string) => actual.isSportEnabled(sport, value()),
  };
});

const COMMITTED: boolean = jest.requireActual('@/lib/sport-switches').HORSE_RACING_ENABLED;

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

function renderFilters() {
  render(
    <>
      <PredictionFilters />
      <RecommendationFilters onFilterChange={jest.fn()} />
    </>
  );
}

beforeEach(() => {
  mockHorseRacing = undefined;
  mockUseRaces.mockReset();
  mockUseRaceDetail.mockReset();
  mockUseRaces.mockReturnValue({ data: [], isLoading: false, error: null, refetch: jest.fn() });
  mockUseRaceDetail.mockReturnValue({ data: undefined, isLoading: true, error: null, refetch: jest.fn() });
});

describe('the committed switch drives every surface', () => {
  it('the Race cards entry is in the nav exactly when the committed constant is true', () => {
    renderNavigation();
    expect(screen.queryAllByRole('link', { name: /race cards/i })).toHaveLength(COMMITTED ? 2 : 0);
  });

  it('/races is paused exactly when the committed constant is false', () => {
    render(<RacesPage />);
    expect(screen.queryByTestId('races-paused') !== null).toBe(!COMMITTED);
    expect(mockUseRaces).toHaveBeenCalledTimes(COMMITTED ? 1 : 0);
  });

  it('the filters offer horse racing exactly when the committed constant is true', () => {
    renderFilters();
    expect(screen.queryAllByRole('option', { name: 'Horse Racing' })).toHaveLength(COMMITTED ? 1 : 0);
    expect(screen.queryAllByRole('option', { name: 'Win (Horse Racing)' })).toHaveLength(COMMITTED ? 1 : 0);
  });
});

describe('horse racing dormant (the constant forced false)', () => {
  beforeEach(() => {
    mockHorseRacing = false;
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

  it('the predictions sport filter and the recommendations market filter leave horse racing out', () => {
    renderFilters();
    expect(screen.queryByRole('option', { name: 'Horse Racing' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Win (Horse Racing)' })).not.toBeInTheDocument();
    // Everything else is still offered.
    for (const sport of ['All Sports', 'Soccer', 'NFL', 'NHL', 'NBA', 'Tennis', 'MMA']) {
      expect(screen.getByRole('option', { name: sport })).toBeInTheDocument();
    }
    for (const market of ['All Markets', '1X2', 'BTTS', 'Moneyline', 'Puck Line']) {
      expect(screen.getByRole('option', { name: market })).toBeInTheDocument();
    }
  });
});

describe('horse racing turned back on (the constant forced true)', () => {
  beforeEach(() => {
    mockHorseRacing = true;
  });

  it('the Race cards entry returns to the sidebar and the header', () => {
    renderNavigation();
    expect(screen.getAllByRole('link', { name: /race cards/i })).toHaveLength(2);
  });

  it('/races fetches and lists race cards again', () => {
    render(<RacesPage />);
    expect(mockUseRaces).toHaveBeenCalledWith({ status: 'scheduled', days_ahead: 2, limit: 200 });
    expect(screen.getByText('No upcoming races')).toBeInTheDocument();
    expect(screen.queryByTestId('races-paused')).not.toBeInTheDocument();
  });

  it('/races/[id] fetches the race again', () => {
    render(<RaceDetailPage params={{ id: 'race-1' }} />);
    expect(mockUseRaceDetail).toHaveBeenCalledWith('race-1');
    expect(screen.queryByTestId('races-paused')).not.toBeInTheDocument();
  });

  it('both filters offer horse racing again', () => {
    renderFilters();
    expect(screen.getByRole('option', { name: 'Horse Racing' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Win (Horse Racing)' })).toBeInTheDocument();
  });
});
