# Descriptive complexity statistics

Median [Q1, Q3] per (model, architecture) cell over the all-completions condition (n = 164 tasks per cell). Lean cluster: Basic, Debugger, AC+Debugger; heavy cluster: AC, ACT, ACT+Debugger.

### gpt-4o

| Architecture | SLOC | CC | Halstead V | Halstead D | Halstead E |
| --- | --- | --- | --- | --- | --- |
| Basic | 5 [2, 8] | 3 [2, 4] | 18 [5, 54] | 1.2 [0.5, 2.5] | 26 [2, 157] |
| AC | 8 [6, 11] | 4 [3, 5] | 40 [12, 73] | 2.0 [0.9, 3.3] | 86 [9, 236] |
| ACT | 8 [5, 11] | 4 [3, 5] | 40 [12, 75] | 2.0 [1.0, 3.3] | 102 [12, 240] |
| Debugger | 5 [2, 8] | 3 [2, 4] | 18 [5, 56] | 1.2 [0.5, 2.5] | 26 [2, 164] |
| AC+Debugger | 5 [2, 8] | 3 [2, 4] | 16 [5, 54] | 1.1 [0.5, 2.6] | 23 [2, 158] |
| ACT+Debugger | 8 [6, 11] | 4 [3, 5] | 38 [12, 84] | 2.0 [1.0, 3.3] | 78 [12, 263] |

### gpt-4o-mini

| Architecture | SLOC | CC | Halstead V | Halstead D | Halstead E |
| --- | --- | --- | --- | --- | --- |
| Basic | 6 [3, 9] | 3 [2, 5] | 27 [5, 58] | 1.5 [0.5, 2.8] | 43 [2, 189] |
| AC | 10 [7, 12] | 4 [3, 6] | 42 [18, 84] | 2.3 [1.3, 3.8] | 102 [24, 268] |
| ACT | 9 [7, 12] | 4 [3, 6] | 49 [18, 83] | 2.3 [1.1, 3.6] | 117 [19, 296] |
| Debugger | 6 [3, 9] | 3 [2, 5] | 27 [5, 59] | 1.5 [0.5, 3.1] | 43 [2, 192] |
| AC+Debugger | 6 [3, 9] | 3 [2, 5] | 27 [5, 60] | 1.5 [0.5, 3.1] | 43 [2, 197] |
| ACT+Debugger | 9 [7, 12] | 5 [3, 6] | 49 [24, 83] | 2.5 [1.5, 3.6] | 123 [36, 300] |

