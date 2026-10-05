### JPM — the buy signal

*Figures as of 2026-10-04, 5-year evidence window. The live table above
recomputes on every run; if it has drifted from the numbers quoted here, the
table is right and this note needs updating.*

**The signal: RSI 34.5 with a 60-day z-score of −2.07, both active now.**

The reason I trust it on *this* name and not as a general rule comes from
three separate panels, in order.

**1. Mean reversion is licensed here (F6).** DFA Hurst is 0.458, and the
rolling estimate has sat around 0.45 for the past year rather than wandering.
That is a mild anti-persistent tilt — moves tend to partially reverse. This
matters because an oversold reading only argues for buying if the series
actually reverts. On a persistent series the same reading argues for the
opposite, which is exactly what happens with AMD. The confidence interval
[0.414, 0.557] does include 0.5, so I am treating this as a weak tilt, not an
established regime.

**2. The rule has actually paid on JPM.** Over the last five years, the 77
days with RSI below 35 were followed by **+4.90% over the next 21 bars, with a
72.7% hit rate**, against a +1.23% average for all other days — an edge of
roughly +3.7 percentage points. The z-score condition is slightly stronger
(+5.47%, 71.8%, n=110). The asymmetry is the convincing part: the mirror-image
overbought condition on the same name returns +0.27% with a 54% hit rate,
i.e. essentially nothing. If both tails printed large numbers I would suspect
I was just measuring JPM's upward drift.

**3. This is a dip, not a dislocation (F5).** Realised 21-day volatility is
19.3% against a 24.4% full-sample level — a ratio of 0.79 — and the GARCH
conditional estimate of 22.7% sits *below* its 25.4% long-run level. Shock
half-life is about 5 days. So price has drifted down on a calm tape rather
than gapping on a volatility expansion. That distinction is the one I care
about most: oversold during a volatility spike usually means the distribution
has changed, and the historical base rate no longer applies.

Supporting context: price is still 3.3% above the 200-day, 9.0% off its high,
and daily VaR(95) is 2.45%.

**What would make me wrong, and one thing that already complicates this.**

The filter I expected to help does not. Splitting the oversold sample by trend
gives +2.21% / 54% when *above* the 200-day (n=63) but +4.48% / 67% when
*below* it (n=108). The "buy the dip in an uptrend" story is the intuitive
one, and on JPM over this window it is the weaker of the two. I am therefore
taking this trade on the mean-reversion evidence alone and not claiming trend
support I cannot actually demonstrate. It also means the current setup sits in
the weaker of the two buckets.

Second caveat: the 21-day forward windows overlap heavily, so those 77
observations are nowhere near 77 independent trials. The true uncertainty is
much wider than the hit rate suggests, which is the same point F8 makes about
walk-forward evaluation.

**Position:** volatility budget allows roughly 26% weight at 1% account risk
with a 2×ATR stop (ATR is 1.94%, so the stop sits 3.9% away).
