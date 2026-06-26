# Post-hoc pairwise comparisons

Matched-pairs rank-biserial correlation r_rb (SLOC; the Holm-significance pattern is identical for all five metrics). Positive r_rb means the first architecture is the more complex. **Bold** entries are significant after Holm correction within the 15-pair family (p < 0.05).

| Comparison | Layer(s) | Type | gpt-4o Primary | gpt-4o Passing | gpt-4o-mini Primary | gpt-4o-mini Passing |
| --- | --- | --- | --- | --- | --- | --- |
| Basic vs AC | R | Single | **-0.90** | **-0.92** | **-0.89** | **-0.89** |
| Basic vs ACT | R,T | Compound | **-0.85** | **-0.86** | **-0.80** | **-0.86** |
| Basic vs Debugger | D | Single | -0.04 | -0.35 | +0.05 | -0.29 |
| Basic vs AC+Debugger | R,D | Compound | -0.31 | -0.44 | +0.09 | +0.15 |
| Basic vs ACT+Debugger | R,T,D | Compound | **-0.94** | **-0.93** | **-0.84** | **-0.89** |
| AC vs ACT | T | Single | +0.30 | +0.13 | +0.13 | +0.05 |
| AC vs Debugger | R↔D | Swap | **+0.85** | **+0.83** | **+0.86** | **+0.87** |
| AC vs AC+Debugger | D | Single | **+0.87** | **+0.86** | **+0.87** | **+0.88** |
| AC vs ACT+Debugger | T,D | Compound | -0.05 | -0.09 | +0.14 | -0.05 |
| ACT vs Debugger | R,T,D | Compound | **+0.78** | **+0.76** | **+0.79** | **+0.84** |
| ACT vs AC+Debugger | T↔D | Swap | **+0.73** | **+0.77** | **+0.80** | **+0.86** |
| ACT vs ACT+Debugger | D | Single | -0.35 | -0.27 | +0.05 | -0.06 |
| Debugger vs AC+Debugger | R | Single | -0.32 | -0.03 | +0.05 | +0.43 |
| Debugger vs ACT+Debugger | R,T | Compound | **-0.89** | **-0.85** | **-0.81** | **-0.85** |
| AC+Debugger vs ACT+Debugger | T | Single | **-0.89** | **-0.87** | **-0.81** | **-0.87** |

