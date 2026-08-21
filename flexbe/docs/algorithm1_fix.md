# Repairing Algorithm 1

## 1. What is wrong

Algorithm 1 has two independent halves:

| lines | content | status |
|---|---|---|
| 5–6 | index formula `ii <- rotate_{n-1}(base, k+1)`, `I^i[0] <- rotate_n(2*ii, k)` | **does not give a conflict-free grouping** |
| 7–11 | control law `R^i = bsm(I^i[0])`, `S^i = n-k-1` on `(n-m)..(n-2)` else `0` | correct, keep as is |

`check_algorithm1.py` shows the failure two ways. Against a validated
conflict-free schedule the printed formula misses at the very first stage for
every size tried (`N` = 8 … 4096, `P_bu` = 2 … 16, both rotation amounts, both
stage orders). And for `N` = 8, 16, 32 with `P_bu` = 2 it enumerates *every*
conflict-free grouping of a stage and finds none compatible: for `N` = 8 the
only conflict-free grouping is the row-based `{0,1,2,3} / {4,5,6,7}`, while
lines 5–6 produce cycle representatives `0` and `2`, which share a row.

Rewriting the printed expression as

```
rotate_n(2*ii, k) = ins( rotate_{n-1}(ii, k), k )
                  = ins( rotate_{n-1}(base, 2k+1), k )
```

(where `ins(x,p)` inserts a zero bit at position `p`) makes the likely origin
visible: the operand is rotated **twice** by roughly the stage index, once in
line 5 and once again by the outer rotation in line 6, and the hole is placed at
bit `k` rather than at `n-1-k`. The rest of the paper needs the hole at `n-1-k`:
that is the only stage order under which line 9 (`S^i = n-k-1`) is the correct
subset-switch state, and it is the order that leaves the result bit reversed,
which is what the Sec. 3.3 write-back path exists to undo.

## 2. The repair

Only lines 5–6 change. Let `m = log2(2*P_bu)`, and

```
ins(x, p) = ((x >> p) << (p+1)) | (x mod 2^p)        insert a 0 bit at position p
```

**Algorithm 1 (revised).** For stage `k` and cycle `j`:

```
h <- n - 1 - k                                  hole bit of this stage

if h < m:                                       both operands share RAM row H
    H <- j
    I^i[2t]   <- (H << m) + ins(t, h)                       t = 0 .. P_bu-1
    S^i       <- h
else:                                           operands in rows H, H + 2^(h-m)
    e <- j mod 2
    H <- ins(j >> 1, h - m)
    I^i[2t]   <- (H << m) + 2t + e                          t = 0 .. P_bu-1
    S^i       <- 0

I^i[2t+1] <- I^i[2t] + 2^h
R^i       <- bsm(I^i[0])                        unchanged, line 7
```

Note that line 9 of the published listing is reproduced exactly: `S^i = h` when
`h < m` is the same as `S^i = n-k-1` for `(n-m) <= k <= (n-2)`, and `S^i = 0`
otherwise. Line 7 is untouched, and `R^i` still equals `bsm(I^i[0])`
— it can also be read straight off the construction as `popcount(H)` (plus `e`
in the second branch), which is cheaper than recomputing `bsm`.

The revised listing also supplies the whole vector `I^i[0 .. 2P_bu-1]`, not only
`I^i[0]`; in the published version the remaining entries are delegated to "a
specific circuit" of [11], which is where a reimplementer has nothing to go on.

## 3. Why it is conflict free

With the shift-down mapping `a_y = bsm(I) = (popcount(I >> m) + (I mod 2^m)) mod 2^m`
one RAM row is a set of `2^m` consecutive indices `I = (H << m) + low`, and its
banks are `(popcount(H) + low) mod 2^m` — the row occupies every bank exactly
once, cyclically shifted by `popcount(H)`.

*Case `h < m`.* Both operands of a butterfly differ in a bit below `m`, so they
lie in the same row. The cycle reads all of row `H`, hence banks
`(popcount(H) + low) mod 2^m` for `low = 0 .. 2^m-1`: all distinct. Slot `j`
holds `low = pi_h[j]`, so slot `j` reads bank `(pi_h[j] + popcount(H)) mod 2^m`
— a fixed stride permutation followed by a rotation, i.e. `P_f = P_s x P_r`
with `S = h`, `R = popcount(H)`. There are `N/2^m` rows, matching the required
`N/(2*P_bu)` cycles per stage.

*Case `h >= m`.* The partner lies in row `H + 2^(h-m)`. Because `H = ins(j>>1, h-m)`
has bit `h-m` equal to zero, `popcount(H + 2^(h-m)) = popcount(H) + 1`. Taking
the `P_bu` low values `2t + e` from both rows gives, for slot `j`, bank

```
    (popcount(H) + e + j) mod 2^m
```

again all distinct — a pure rotation, so `S = 0` and `R = (popcount(H)+e) mod 2^m`.
Halving the rows (only those with bit `h-m` clear) and doubling for `e ∈ {0,1}`
again yields exactly `N/2^m` cycles.

Both branches also visit each of the `N` indices exactly once per stage, and
`I^i[2t]` always has bit `h` clear, so `I^i[2t+1] = I^i[2t] + 2^h` is its
partner. `test_flexbe.TestAlgorithm1Fix` asserts all of this directly from the
formula, for `N` = 8 … 4096 and `P_bu` = 1 … 16, and runs an engine whose
control comes only from the revised listing against `numpy.fft`.

## 4. Cost, and what is still free

Per cycle the revised generator needs an `(n-m)`-bit cycle counter, one
insert-a-zero network at a stage-constant position (one 2:1 mux per bit, the
select being a thermometer code of `h-m`, so it folds into the existing barrel
shifter control), and one popcount of the `(n-m)`-bit row index — which the
`bsm` datapath already computes. `popcount(H)` can also be maintained
incrementally across `j`. This is still far lighter than the priority-encoder
reverse lookup of BE-base described in Sec. 5.3.2, so the Fig. 10 comparison is
unaffected.

The *order* of cycles within a stage carries no functional meaning — only the
grouping does. If a rotation-style counter is preferred for the RTL, any
bijection from `j` to `(H, e)` may replace the one above; the essential content
of the repair is the grouping rule and the placement of the hole at `h = n-1-k`.

## 5. LaTeX

`algorithm1_fixed.tex` contains a drop-in `algorithm2e` listing.
