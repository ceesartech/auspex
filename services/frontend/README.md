# CeesarBet Predict - Frontend

Next.js 14 frontend for the CeesarBet Predict sports betting analytics system.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS 3
- **State**: Zustand + React Query (TanStack)
- **Charts**: Recharts
- **Forms**: react-hook-form + Zod
- **Testing**: Jest + Playwright

## Quick Start

```bash
npm install
npm run dev        # Development server at http://localhost:3000
npm run build      # Production build
npm test           # Unit tests
npm run test:e2e   # E2E tests (requires running dev server)
```

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── (authenticated)/    # Auth-guarded routes
│   │   ├── page.tsx        # Dashboard
│   │   ├── predictions/    # Prediction pages
│   │   ├── recommendations/
│   │   ├── accumulator/
│   │   ├── analytics/
│   │   ├── lottery/
│   │   └── settings/
│   └── login/
├── components/
│   ├── ui/                 # Base UI components
│   ├── layout/             # Header, Sidebar, Footer
│   ├── shared/             # Loading, Error, Empty states
│   ├── providers/          # Theme, Query providers
│   ├── dashboard/          # Dashboard widgets
│   ├── predictions/        # Prediction components
│   ├── recommendations/    # Recommendation components
│   ├── accumulator/        # Accumulator builder
│   └── charts/             # Recharts wrappers
├── lib/
│   ├── api/                # Axios API client
│   ├── hooks/              # React Query hooks
│   ├── store/              # Zustand stores
│   ├── types/              # TypeScript interfaces
│   └── utils/              # Format, validation, calculations
└── styles/
    └── globals.css         # Tailwind + CSS variables
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000` | WebSocket URL |

## Features

- Real-time predictions via WebSocket
- Interactive accumulator builder
- Model performance analytics with charts
- Lottery number analysis
- Dark/light theme support
- PWA-ready with manifest
- Responsive design (mobile-first)
