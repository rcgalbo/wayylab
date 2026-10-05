### What the dashboard is telling me right now

*As of 2026-10-04. Three names, one oversold, one overbought, one deeply
oversold — and the correct action is different in each case.*

| | JPM | AMD | TLT |
|---|---|---|---|
| RSI 14 | 34.5 | 70.1 | 23.5 |
| Naive read | buy | sell | strong buy |
| **My call** | **buy** | **trim** | **avoid** |
| DFA Hurst | 0.458 | 0.556 | 0.503 |
| Historical edge on the active signal | +3.7pp | +8.4pp *(bullish)* | −1.4pp |
| Ann. vol | 24.4% | 56.9% | 15.7% |
| Vol budget weight | 26% | 13% | 47% *(not a recommendation)* |

**The buy: JPM, oversold at RSI 34.5 / z −2.07.** Hurst below 0.5 says
reversion is the right model for this series; the conditional table says the
rule has returned +4.90% over 21 days with a 72.7% hit rate against a +1.23%
baseline; and GARCH says current volatility is *below* its long-run level, so
this is a drift lower on a calm tape rather than a repricing. Full reasoning
and the one piece of evidence that cuts against it are on the JPM tab.

**The sell: AMD, trim on volatility budget — not on RSI.** This is the part I
would defend hardest. The textbook sell signal is present and the data says it
is wrong: RSI>70 on AMD has been followed by +11.67% with a 63% hit rate, an
edge of +8.4 points, consistent with its Hurst of 0.556. So I am not selling
because it is extended. I am selling because 56.9% annualised volatility, a
5.6% daily VaR, a −65% historical drawdown and a 39-day volatility half-life
mean the position has outgrown its risk budget — 13% weight supportable against
a position that has run 69% above its 200-day. Keep the thesis, cut the size.

**The control: TLT.** The most oversold name on the page and the one I will not
touch, because the signal that would justify buying it has a negative track
record here (−0.37% after RSI<35, 46% hit) and there is no drift to revert
toward — Sharpe is −0.80 and GARCH persistence of 0.998 says there isn't even a
stable volatility level to anchor on.

**The method, which is the actual point.** No indicator reading carries a
recommendation on its own. Each of the three decisions came from the same
three-step test: does the series' structure license this model (Hurst), has
this specific rule ever paid on this specific name against a baseline
(conditional table), and does the current volatility regime resemble the
history I am averaging over (GARCH)? That is the same discipline F8 applies to
forecasting, where ARIMA fails to beat a random walk and the dashboard says so
rather than hiding it. A signal that has not been compared against a baseline
is not a signal.

**Known weaknesses.** The 21-day forward windows overlap, so effective sample
sizes are far smaller than the n column implies and none of these edges carry
a significance test. Five years covers one broad market regime. Every
conditional edge here is in-sample on the same window used to choose the names.
