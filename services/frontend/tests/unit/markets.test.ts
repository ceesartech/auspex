import { HIDDEN_SOCCER_MARKETS, marketOptionsBySport } from '@/lib/markets';

describe('market dropdown options', () => {
  const soccerValues = marketOptionsBySport.soccer.map((o) => o.value);

  it('hides odd_even (lost to a per-league constant) and btts (operator decision)', () => {
    // services/api prediction_service.HIDDEN_PREDICTION_PAIRS holds 'ensemble:odd_even';
    // an api unit test checks this list against it.
    expect(HIDDEN_SOCCER_MARKETS).toEqual(['btts', 'odd_even']);
  });

  it('never offers a hidden soccer market', () => {
    for (const hidden of HIDDEN_SOCCER_MARKETS) {
      expect(soccerValues).not.toContain(hidden);
    }
  });

  it('keeps every other soccer market', () => {
    expect(soccerValues).toEqual([
      '',
      'over_under',
      'asian_handicap',
      'correct_score',
      'double_chance',
      'draw_no_bet',
      'total_goals',
      'winning_margin',
      'team_total',
      'clean_sheet',
      'win_to_nil',
      'over_under_ht',
      'match_result_ht',
      'btts_ht',
      'result_btts',
      'result_over_under',
      'ht_ft_double_result',
    ]);
    expect(new Set(soccerValues).size).toBe(soccerValues.length);
  });

  it('leaves the other sports untouched', () => {
    expect(marketOptionsBySport.nhl.map((o) => o.value)).toEqual(['', 'moneyline', 'regulation', 'puck_line', 'total']);
    expect(marketOptionsBySport.nba.map((o) => o.value)).toEqual(['', 'moneyline', 'spread', 'total']);
    expect(marketOptionsBySport.nfl.map((o) => o.value)).toEqual(['', 'moneyline', 'spread', 'total']);
    expect(Object.keys(marketOptionsBySport).sort()).toEqual(['nba', 'nfl', 'nhl', 'soccer']);
  });
});
