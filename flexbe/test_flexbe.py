"""
test_flexbe.py -- verification suite for the FlexBE / BSPNet simulator
=====================================================================

Run with

    python test_flexbe.py            # or:  python -m unittest -v test_flexbe

The tests check the architectural claims of the paper, not just the numerics:

  * the shift-down mapping of Eq. (3) is bank-conflict free for every cycle of
    every stage, and each stage touches every element exactly once,
  * the PRS decomposition P_f = P_s x P_r reproduces the FCS crossbar cycle by
    cycle, R^i = bsm(I^i[0]) and S^i matches the control law of Algorithm 1,
  * the datapath computes an exact FFT (natural order after the integrated
    bit-reversal write-back) and an exact butterfly-linear layer,
  * sub-parallelism folds 2*P_bu/l short transforms into one pass (Alg. 2),
  * Algorithm 3 realises the global bit-reversal permutation, conflict free on
    both memories, in exactly N/P cycles,
  * measured cycle counts agree with Eq. (10), including the headline
    "four 32k-point FFTs in 15,360 cycles" of the abstract.
"""

import math
import unittest

import numpy as np

import flexbe as fx
import bspnet as bs


# ---------------------------------------------------------------------------
class TestBitHelpers(unittest.TestCase):

    def test_bit_rev(self):
        self.assertEqual(fx.bit_rev(1, 3), 4)          # 001 -> 100, Sec. 3.3
        self.assertEqual(fx.bit_rev(0b1011, 4), 0b1101)
        for n in (1, 3, 5, 8):
            v = fx.bit_rev_array(n)
            self.assertTrue(np.array_equal(v[v], np.arange(1 << n)))

    def test_rotl(self):
        self.assertEqual(fx.rotl(0b100, 1, 3), 0b001)
        self.assertEqual(fx.rotl(0b100, 2, 3), 0b010)
        self.assertEqual(fx.rotl(0b1011, 4, 4), 0b1011)

    def test_bsm_matches_vector_version(self):
        for m in (1, 2, 4, 5):
            idx = np.arange(1 << (m + 3))
            ref = np.array([fx.bsm(int(i), m) for i in idx])
            self.assertTrue(np.array_equal(ref, fx.bsm_array(idx, m)))

    def test_shift_down_is_a_bijection(self):
        """Every element must own a unique (bank, depth) slot."""
        for n, m in [(3, 2), (5, 2), (8, 4), (10, 5)]:
            idx = np.arange(1 << n)
            slots = fx.bsm_array(idx, m) * (1 << (n - m)) + (idx >> m)
            self.assertEqual(len(np.unique(slots)), 1 << n)


# ---------------------------------------------------------------------------
class TestSchedule(unittest.TestCase):
    """Conflict freedom, coverage and the PRS/FCS equivalence."""

    CASES = [(3, 1), (4, 2), (5, 2), (6, 4), (8, 4), (10, 16), (7, 1)]

    def test_conflict_free_and_complete(self):
        for n, P_bu in self.CASES:
            m = (2 * P_bu).bit_length() - 1
            if n < m:
                continue
            with self.subTest(N=1 << n, P_bu=P_bu):
                sched = fx.ButterflySchedule(n, m, validate=True)
                self.assertEqual(sched.total_cycles,
                                 (1 << n) // (2 * P_bu) * n)

    def test_prs_equals_fcs(self):
        n, P_bu = 8, 4
        m = (2 * P_bu).bit_length() - 1
        prs = fx.PermuteRotateSwitch(m)
        sched = fx.ButterflySchedule(n, m)
        rng = np.random.default_rng(0)
        for ctrl in sched.all_cycles():
            D = rng.normal(size=1 << m) + 1j * rng.normal(size=1 << m)
            Pf = fx.FullyConnectedSwitch.matrix(ctrl.indices, m)
            self.assertTrue(np.allclose(prs.forward(D, ctrl.R, ctrl.S),
                                        fx.FullyConnectedSwitch.forward(D, Pf)))
            # PRS-B undoes PRS-A (orthogonality of the permutation matrices)
            X = prs.forward(D, ctrl.R, ctrl.S)
            self.assertTrue(np.allclose(prs.inverse(X, ctrl.R, ctrl.S), D))

    def test_control_matches_algorithm1(self):
        """S^i = n-k-1 for (n-m) <= k <= n-2 else 0;  R^i = bsm(I^i[0])."""
        for n, P_bu in [(6, 4), (8, 4), (10, 16), (12, 16)]:
            m = (2 * P_bu).bit_length() - 1
            sched = fx.ButterflySchedule(n, m)
            for k, stage in enumerate(sched.stages):
                expect = sched.paper_subset_state(k)
                for ctrl in stage:
                    self.assertEqual(ctrl.S, expect)
                    self.assertEqual(ctrl.R, fx.bsm(int(ctrl.indices[0]), m))

    def test_fixed_algorithm1_reproduces_the_schedule(self):
        for n, P_bu in [(3, 1), (4, 2), (6, 4), (8, 4), (10, 16), (5, 16)]:
            m = (2 * P_bu).bit_length() - 1
            if n < m:
                continue
            with self.subTest(N=1 << n, P_bu=P_bu):
                sched = fx.ButterflySchedule(n, m, validate=True)
                for k, stage in enumerate(sched.stages):
                    for j, c in enumerate(stage):
                        f = fx.algorithm1_fixed(n, P_bu, k, j)
                        self.assertTrue(np.array_equal(f.indices, c.indices))
                        self.assertTrue(np.array_equal(f.depth_by_bank,
                                                       c.depth_by_bank))
                        self.assertEqual((f.R, f.S), (c.R, c.S))

    def test_subset_switch_uses_only_m_states(self):
        """Sec. 3.1: P_s takes only log2(2*P_bu) of the N/(2P_bu) possible P_f."""
        n, P_bu = 12, 16
        m = (2 * P_bu).bit_length() - 1
        sched = fx.ButterflySchedule(n, m)
        states = {c.S for c in sched.all_cycles()}
        self.assertLessEqual(len(states), m)
        self.assertTrue(states <= set(range(m)))


# ---------------------------------------------------------------------------
class TestAlgorithm1Fix(unittest.TestCase):
    """The repaired lines 5-6, verified without reference to the schedule."""

    CASES = [(3, 1), (4, 2), (6, 2), (6, 4), (9, 4), (10, 16), (12, 16), (5, 16)]

    def _cases(self):
        for n, P_bu in self.CASES:
            if n >= (2 * P_bu).bit_length() - 1:
                yield n, P_bu

    def test_conflict_free_and_complete(self):
        """Directly from the formula: distinct banks, exact coverage, valid pairs."""
        for n, P_bu in self._cases():
            m = (2 * P_bu).bit_length() - 1
            N, P = 1 << n, 1 << m
            with self.subTest(N=N, P_bu=P_bu):
                for k in range(n):
                    seen = np.zeros(N, dtype=bool)
                    h = n - 1 - k
                    for j in range(N // P):
                        c = fx.algorithm1_fixed(n, P_bu, k, j)
                        banks = fx.bsm_array(c.indices, m)
                        self.assertEqual(len(np.unique(banks)), P,
                                         "bank conflict")
                        self.assertFalse(seen[c.indices].any(), "element reused")
                        seen[c.indices] = True
                        # correct butterfly pairing on the hole bit
                        self.assertTrue(np.all(c.indices[1::2] - c.indices[0::2]
                                               == (1 << h)))
                        self.assertFalse(np.any((c.indices[0::2] >> h) & 1))
                    self.assertTrue(seen.all(), "stage does not cover all N")

    def test_lines_7_to_11_are_unchanged(self):
        """R^i = bsm(I^i[0]) and S^i = n-k-1 on (n-m)..(n-2), else 0."""
        for n, P_bu in self._cases():
            m = (2 * P_bu).bit_length() - 1
            for k in range(n):
                expect_S = (n - k - 1) if (n - m) <= k <= (n - 2) else 0
                for j in range(1 << (n - m)):
                    c = fx.algorithm1_fixed(n, P_bu, k, j)
                    self.assertEqual(c.R, fx.bsm(int(c.indices[0]), m))
                    self.assertEqual(c.S, expect_S)

    def test_prs_still_replaces_the_crossbar(self):
        """P_f = P_s x P_r still holds for every cycle the formula emits."""
        n, P_bu = 10, 16
        m = (2 * P_bu).bit_length() - 1
        prs = fx.PermuteRotateSwitch(m)
        for k in range(n):
            for j in range(1 << (n - m)):
                c = fx.algorithm1_fixed(n, P_bu, k, j)
                self.assertTrue(np.array_equal(prs.read_map(c.R, c.S),
                                               fx.bsm_array(c.indices, m)))

    def test_engine_driven_only_by_the_fixed_algorithm(self):
        """An engine whose control comes solely from the repaired listing."""
        for N, P_bu in [(64, 4), (1024, 16)]:
            with self.subTest(N=N, P_bu=P_bu):
                n = int(math.log2(N))
                eng = fx.FlexBE(P_bu)
                eng._sched[(n, 1)] = fx.ButterflySchedule.from_algorithm1_fixed(
                    n, P_bu, validate=True)
                rng = np.random.default_rng(N)
                x = rng.normal(size=N) + 1j * rng.normal(size=N)
                y, st = eng.transform(x, bitrev=True, datapath="cycle")
                self.assertTrue(np.allclose(y[0], np.fft.fft(x)))
                self.assertEqual(st.butterfly_cycles, N // (2 * P_bu) * n)

    def test_published_formula_is_not_conflict_free(self):
        """Regression guard for the diagnosis: lines 5-6 as printed do fail."""
        def published_I0(n, P_bu, k, j, rot=1):
            return fx.rotl(2 * fx.rotl(j * P_bu, k + rot, n - 1), k, n)

        n, P_bu = 10, 16
        m = (2 * P_bu).bit_length() - 1
        sched = fx.ButterflySchedule(n, m)
        gid = np.zeros(1 << n, dtype=int)
        for g, c in enumerate(sched.stages[0]):
            gid[c.indices] = g
        reps = [published_I0(n, P_bu, 0, j) for j in range(1 << (n - m))]
        self.assertNotEqual(sorted(gid[reps]), list(range(len(sched.stages[0]))))
        # while the repaired formula does select one representative per cycle
        fixed = [int(fx.algorithm1_fixed(n, P_bu, 0, j).indices[0])
                 for j in range(1 << (n - m))]
        self.assertEqual(sorted(gid[fixed]), list(range(len(sched.stages[0]))))


# ---------------------------------------------------------------------------
class TestFFT(unittest.TestCase):

    def _fft_case(self, N, P_bu, datapath, fmt=None, stage_scale=None):
        rng = np.random.default_rng(N + P_bu)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        x /= 4 * np.max(np.abs(x))
        eng = fx.FlexBE(P_bu, fmt=fmt)
        y, st = eng.transform(x, bitrev=True, datapath=datapath,
                              stage_scale=stage_scale, validate=True)
        ref = np.fft.fft(x) * st.scale
        return y[0], ref, st

    def test_fft_matches_numpy(self):
        for N, P_bu in [(32, 2), (64, 4), (256, 4), (1024, 16), (4096, 16)]:
            with self.subTest(N=N, P_bu=P_bu):
                y, ref, st = self._fft_case(N, P_bu, "cycle")
                self.assertLess(np.max(np.abs(y - ref)) / np.max(np.abs(ref)),
                                1e-12)

    def test_cycle_and_vector_datapaths_agree(self):
        yc, _, _ = self._fft_case(256, 4, "cycle")
        yv, _, _ = self._fft_case(256, 4, "vector")
        self.assertTrue(np.allclose(yc, yv))

    def test_output_is_bit_reversed_without_writeback(self):
        """DIF with natural-order input leaves the result bit reversed."""
        N, P_bu = 256, 4
        rng = np.random.default_rng(3)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        eng = fx.FlexBE(P_bu)
        y, _ = eng.transform(x, bitrev=False, datapath="cycle")
        rev = fx.bit_rev_array(int(math.log2(N)))
        self.assertTrue(np.allclose(y[0][rev], np.fft.fft(x)))

    def test_packing_multiple_sequences(self):
        """P_N sequences share one banked RAM array (Sec. 3.4, Fig. 5)."""
        N, P_bu, P_N = 128, 4, 5
        rng = np.random.default_rng(7)
        x = rng.normal(size=(P_N, N)) + 1j * rng.normal(size=(P_N, N))
        eng = fx.FlexBE(P_bu)
        y, st = eng.transform(x, bitrev=True, datapath="cycle")
        self.assertTrue(np.allclose(y, np.fft.fft(x, axis=1)))
        self.assertEqual(st.ram_depth, P_N * N // (2 * P_bu))
        self.assertEqual(st.butterfly_cycles,
                         P_N * N // (2 * P_bu) * int(math.log2(N)))

    def test_inverse_transform(self):
        N, P_bu = 64, 4
        rng = np.random.default_rng(11)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        eng = fx.FlexBE(P_bu)
        X, _ = eng.transform(x, bitrev=True, datapath="cycle")
        C = fx.fft_coefficients(N, inverse=True)
        y, _ = eng.transform(X[0], C, bitrev=True, datapath="cycle")
        self.assertTrue(np.allclose(y[0] / N, x))


# ---------------------------------------------------------------------------
class TestSubParallel(unittest.TestCase):
    """Sec. 3.2 / Algorithm 2: transforms shorter than 2*P_bu."""

    def test_geometry(self):
        eng = fx.FlexBE(16)
        self.assertEqual(eng.geometry(8), (32, 4))     # Fig. 3 example, P_sub=4
        self.assertEqual(eng.geometry(32), (32, 1))
        self.assertEqual(eng.geometry(1024), (1024, 1))

    def test_short_ffts(self):
        for P_bu, l, n_seq in [(4, 4, 6), (4, 2, 8), (16, 8, 13), (16, 4, 4)]:
            with self.subTest(P_bu=P_bu, l=l):
                rng = np.random.default_rng(l * 31 + P_bu)
                x = rng.normal(size=(n_seq, l)) + 1j * rng.normal(size=(n_seq, l))
                eng = fx.FlexBE(P_bu)
                y, st = eng.transform(x, bitrev=True, datapath="cycle")
                self.assertEqual(st.P_sub, 2 * P_bu // l)
                self.assertEqual(st.n_stages, int(math.log2(l)))
                self.assertTrue(np.allclose(y, np.fft.fft(x, axis=1)))

    def test_interleaving_is_eq5(self):
        eng = fx.FlexBE(4)                       # P = 8
        x = np.array([[0, 1, 2, 3], [10, 11, 12, 13]])
        comp = eng.interleave(x, P_sub=2)
        self.assertTrue(np.array_equal(comp[0],
                                       [0, 10, 1, 11, 2, 12, 3, 13]))
        back = eng.deinterleave(comp, l=4, P_sub=2, n_seq=2)
        self.assertTrue(np.array_equal(back, x))

    def test_subparallel_saves_cycles_versus_zero_padding(self):
        """FlexBE folds 2P_bu/l transforms per pass; BE-base zero pads."""
        P_bu, l, n_seq = 16, 8, 4096
        flex = fx.layer_cycles(n_seq, l, P_be=4, P_bu=P_bu)
        padded = fx.layer_cycles(n_seq, 2 * P_bu, P_be=4, P_bu=P_bu)
        self.assertAlmostEqual(padded / flex, (2 * P_bu / l) * (5 / 3), places=6)


# ---------------------------------------------------------------------------
class TestBitReversal(unittest.TestCase):
    """Sec. 3.3 and Algorithm 3."""

    def test_algorithm3_is_a_conflict_free_permutation(self):
        for n, m in [(4, 2), (6, 2), (10, 5), (12, 5), (15, 5), (8, 4)]:
            with self.subTest(N=1 << n, P=1 << m):
                j, k = fx.bitrev_schedule(n, m)
                P, N = 1 << m, 1 << n
                self.assertEqual(j.shape, (N // P, P))     # N/P cycles exactly
                self.assertEqual(len(np.unique(j)), N)     # covers the index space
                self.assertTrue(np.array_equal(k, fx.bit_rev_array(n)[j]))
                for jv, kv in zip(j, k):
                    src = fx.bsm_array(jv, m)              # FlexBE RAM banks
                    dst = kv & (P - 1)                     # global buffer banks
                    self.assertEqual(len(np.unique(src)), P)
                    self.assertEqual(len(np.unique(dst)), P)
                    # lane a always reads bank a  (Sec. 3.3.2)
                    self.assertTrue(np.array_equal(src, np.arange(P)))

    def test_eq7_destination_bank_law(self):
        """A^gb_y[a] = bit-rev_m((a + u) mod P): rotation then bit reversal."""
        n, m = 12, 5
        P = 1 << m
        j, k = fx.bitrev_schedule(n, m)
        brev_m = fx.bit_rev_array(m)
        cyc = 0
        for u in range(P):
            expect = brev_m[(np.arange(P) + u) % P]
            for _ in range(j.shape[0] // P):
                self.assertTrue(np.array_equal(k[cyc] & (P - 1), expect))
                cyc += 1

    def test_bitrev_cycle_count(self):
        N, P_bu = 1024, 16
        rng = np.random.default_rng(5)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        eng = fx.FlexBE(P_bu)
        _, st = eng.transform(x, bitrev=True, datapath="cycle")
        self.assertEqual(st.bitrev_cycles, N // (2 * P_bu))


# ---------------------------------------------------------------------------
class TestButterflyLinear(unittest.TestCase):
    """The same datapath with trained weights instead of twiddles."""

    def test_bl_matches_dense_matrix(self):
        l, P_bu = 32, 16
        rng = np.random.default_rng(2)
        C = fx.random_bl_coefficients(l, rng)
        eng = fx.FlexBE(P_bu)
        # build the equivalent dense matrix by transforming the identity
        eye, _ = eng.transform(np.eye(l), C, datapath="cycle")
        M = np.asarray(eye).T
        x = rng.normal(size=(7, l))
        y, _ = eng.transform(x, C, datapath="cycle")
        self.assertTrue(np.allclose(y, x @ M.T))

    def test_bl_matches_reference_model(self):
        for l, P_bu in [(16, 4), (32, 16), (8, 16), (64, 8)]:
            with self.subTest(l=l, P_bu=P_bu):
                rng = np.random.default_rng(l)
                C = fx.random_bl_coefficients(l, rng)
                eng = fx.FlexBE(P_bu)
                x = rng.normal(size=(4, l))
                y, _ = eng.transform(x, C, datapath="cycle")
                N, P_sub = eng.geometry(l)
                comp = eng.interleave(x.astype(complex), P_sub)
                ref = fx.butterfly_reference(comp, fx.expand_coefficients(C, P_sub),
                                             P_sub)
                ref = eng.deinterleave(ref, l, P_sub, 4)
                self.assertTrue(np.allclose(y, ref))

    def test_paper_weight_layout(self):
        """[2, 2, l/2, log2 l] of Table 4 -> internal (log2 l, l/2, 2, 2)."""
        l = 16
        rng = np.random.default_rng(1)
        W = rng.normal(size=(2, 2, l // 2, int(math.log2(l))))
        C = fx.coefficients_from_paper_layout(W)
        self.assertEqual(C.shape, (int(math.log2(l)), l // 2, 2, 2))
        self.assertEqual(C[3, 5, 1, 0], W[1, 0, 5, 3])


# ---------------------------------------------------------------------------
class TestFixedPoint(unittest.TestCase):
    """Sec. 5.2: Q1.15, 32-bit accumulation, convergent rounding, saturation."""

    def test_quantiser(self):
        fmt = fx.FixedPointFormat()
        self.assertAlmostEqual(fmt.max_val, 1.0 - 2 ** -15, places=9)
        self.assertEqual(fmt.min_val, -1.0)
        self.assertEqual(fmt.quantise(2.0), fmt.max_val)      # saturation
        self.assertEqual(fmt.quantise(-5.0), fmt.min_val)
        self.assertAlmostEqual(float(fmt.quantise(0.5 + 2 ** -16)), 0.5, places=9)

    def test_fxp16_fft_accuracy(self):
        N, P_bu = 1024, 16
        fmt = fx.FixedPointFormat()
        rng = np.random.default_rng(9)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        x = fmt.quantise(x / (4 * np.max(np.abs(x))))
        eng = fx.FlexBE(P_bu, fmt=fmt)
        y, st = eng.transform(x, bitrev=True, stage_scale=0.5, datapath="cycle")
        ref = np.fft.fft(x) * st.scale
        err = np.linalg.norm(y[0] - ref) / np.linalg.norm(ref)
        self.assertLess(err, 1e-2)
        snr_db = 20 * math.log10(1 / err)
        self.assertGreater(snr_db, 40.0)


# ---------------------------------------------------------------------------
class TestPerformanceModel(unittest.TestCase):

    def test_measured_cycles_match_eq10(self):
        for N, P_bu, P_N in [(64, 4, 3), (256, 16, 2), (32, 16, 8), (8, 16, 16)]:
            with self.subTest(N=N, P_bu=P_bu):
                eng = fx.FlexBE(P_bu)
                rng = np.random.default_rng(N)
                x = rng.normal(size=(P_N, N))
                _, st = eng.transform(x, datapath="cycle")
                model = fx.cycles_eq10(P_N, st.N, st.P_sub, 1, P_bu)
                self.assertAlmostEqual(st.butterfly_cycles, model, places=6)

    def test_abstract_headline_15360_cycles(self):
        """'four 32k-point FFTs in approximately 15,360 clock cycles'."""
        self.assertEqual(fx.cycles_eq10(4, 32768, 1, P_be=4, P_bu=16), 15360)
        arr = fx.BEArray(P_be=4, P_bu=16)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 32768)) + 1j * rng.normal(size=(4, 32768))
        y, st = arr.transform(x, bitrev=False, datapath="vector")
        self.assertEqual(st.butterfly_cycles, 15360)
        self.assertTrue(np.allclose(y, np.fft.fft(x, axis=1)[:, fx.bit_rev_array(15)]))

    def test_prs_cheaper_than_fcs(self):
        """Fig. 9: the PRS advantage grows with the number of switching ports."""
        prev = 0.0
        for P_bu in (4, 8, 16, 32, 64):               # 8..128 ports
            c = fx.interconnect_cost(P_bu)
            self.assertLess(c["prs_mux2"], c["fcs_mux2"])
            self.assertGreater(c["ratio"], prev)      # gap widens with P_bu
            prev = c["ratio"]
        # a 4-port switch is the crossover: the staged PRS has no advantage yet
        self.assertGreaterEqual(fx.interconnect_cost(2)["prs_mux2"],
                                fx.interconnect_cost(2)["fcs_mux2"])
        # quadratic (FCS) versus linear-in-m*P (PRS) scaling over 8x more ports
        c8, c64 = fx.interconnect_cost(8), fx.interconnect_cost(64)
        self.assertGreater(c64["fcs_mux2"] / c8["fcs_mux2"], 60.0)   # ~ 8**2
        self.assertLess(c64["prs_mux2"] / c8["prs_mux2"], 16.0)      # ~ 8*m/m'


# ---------------------------------------------------------------------------
class TestBSPNet(unittest.TestCase):

    def test_feature_extraction_matches_numpy(self):
        cfg = bs.BSPNetConfig("small", d_in=8, d_m=32, L=32,
                              n_mlp_blocks=2, bit_reversal=True, N=1024)
        net = bs.BSPNet(cfg, bs.HW["FlexBE"], seed=0)
        rng = np.random.default_rng(4)
        s = (rng.normal(size=cfg.N) + 1j * rng.normal(size=cfg.N)) / 8
        got = net.features(s)
        ref = net.features_reference(s)
        for key in ref:
            self.assertTrue(np.allclose(got[key], ref[key], atol=1e-8),
                            msg=f"{key} mismatch")

    def test_forward_shape_and_determinism(self):
        cfg = bs.BSPNetConfig("small", d_in=8, d_m=32, L=64,
                              n_mlp_blocks=2, bit_reversal=True, N=1024)
        net = bs.BSPNet(cfg, bs.HW["FlexBE"], seed=1)
        rng = np.random.default_rng(6)
        s = (rng.normal(size=cfg.N) + 1j * rng.normal(size=cfg.N)) / 8
        logits = net.forward(s)
        self.assertEqual(logits.shape, (cfg.n_classes,))
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertGreater(net.cycles, 0)
        self.assertTrue(np.allclose(logits, net.forward(s)))

    def test_cfg6_latency_is_plausible(self):
        cfg = bs.CONFIGS["cfg-6"]
        lat = bs.latency_ms(cfg, bs.HW["BSP-Flex"])
        self.assertGreater(lat, 0.05)
        self.assertLess(lat, 0.5)                     # Fig. 11: ~0.2 ms
        thr = bs.throughput_sps(cfg, bs.HW["BSP-Flex"])
        self.assertGreater(thr, 2000)                 # Fig. 15a peak ~4.5k/s

    def test_flexbe_beats_be_base_on_every_config(self):
        for name, cfg in bs.CONFIGS.items():
            with self.subTest(cfg=name):
                flex = bs.latency_ms(cfg, bs.HW["FlexBE"])
                for base in ("BE-base-1", "BE-base-2"):
                    self.assertLess(flex, bs.latency_ms(cfg, bs.HW[base]),
                                    msg=f"{name} vs {base}")

    def test_fft_dominates_only_for_large_din(self):
        cfg = bs.CONFIGS["cfg-6"]
        items = dict(bs.cycle_breakdown(cfg, bs.HW["FlexBE"]))
        fft = [v for k, v in items.items() if "FFT" in k][0]
        self.assertEqual(fft, 15360)


if __name__ == "__main__":
    unittest.main(verbosity=2)
