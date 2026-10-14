---
name: algebraic-equation-solving
description: Use when a question involves solving linear or quadratic equations for an unknown.
---
## Equation Solving Procedure
1. Move all terms containing the unknown to the left-hand side and constants to the right.
2. For linear equations `ax + b = c`, compute `x = (c - b) / a`.
3. For quadratics `ax^2 + bx + c = 0`, compute the discriminant `d = b^2 - 4ac`; if `d >= 0`,
   return `x = (-b ± sqrt(d)) / (2a)`.
4. Substitute each root back into the original equation to verify it.

## Do NOT emit
- roots that fail the substitution check
