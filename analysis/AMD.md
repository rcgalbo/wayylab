### AMD — the sell signal, and why the obvious reason for it is wrong

*Figures as of 2026-10-04, 5-year evidence window.*

**The obvious read, which I think is a trap.** RSI is 70.1, the 60-day
z-score is +2.19, price is at its high and 69% above the 200-day. Every
textbook overbought box is ticked. The generic conclusion is "sell."

**The data on this name says the opposite.** The 151 days where AMD's RSI
exceeded 70 were followed by **+11.67% over the next 21 bars, with a 62.9% hit
rate**, against a +3.28% baseline — an edge of +8.4 percentage points. Being
stretched upward (z > +1.5, n=245) returns +7.19%. On AMD, overbought has been
one of the better things that can happen to the chart.

That is consistent with F6: DFA Hurst is 0.556 and the rolling estimate is
0.590, the opposite tilt to JPM. On a persistent series, extension tends to
continue, and fading it is fighting the series' own structure. A trader who
applied the RSI>70 rule mechanically here would have been run over repeatedly.

**So the sell signal is not a momentum call — it is a risk call.**

The sell argument rests entirely on the risk panel, and it is sufficient on
its own:

- Annualised volatility is **56.9%**, against JPM's 24.4%.
- Daily VaR(95) is **5.60%** and CVaR(95) is 7.72%. A bad day here is a bad
  quarter in JPM.
- Maximum drawdown over the window is **−65.4%**.
- GARCH persistence is **0.983 with a 39-day shock half-life**, against 5 days
  for JPM. When volatility rises on this name it does not come back quickly,
  so the exposure I am holding is not the exposure I will be holding if the
  regime turns.

The concrete trigger is position size, not price. At 1% account risk with a
2×ATR stop, AMD's 3.82% ATR implies a maximum weight near **13%** — roughly
half the 26% that JPM's volatility supports. A position opened at a sensible
size that has since run 69% above its 200-day is now, mechanically, well
beyond its volatility budget.

**The call: trim back to the volatility budget, do not exit.** I am not
forecasting a decline — the conditional evidence argues the opposite, and I
would be contradicting my own table if I claimed otherwise. I am saying that
the expected return no longer compensates for the size of the position, and
that the honest action on a +8pp-edge signal attached to a −65% drawdown
history is to keep the exposure and cut the quantity.

**What would change this.** If 21-day realised volatility pushed meaningfully
above the 56.9% full-sample level, or the GARCH conditional estimate rose
above its 59.6% long-run level (it is currently just below, at 58.3%), the
regime read changes and the trim becomes an exit. Right now volatility is
elevated but not expanding — ratio 0.93 — which is why this is a sizing
decision rather than a directional one.
