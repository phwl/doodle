"""
test_flexbe.py -- verification suite
====================================

    python test_flexbe.py            # or: python -m unittest -v test_flexbe

The tests check architectural claims, not just numerics:

  * the shift-down mapping is bank-conflict free for every cycle of every
    stage, each stage covers every element once, and the PRS reproduces the
    FCS crossbar cycle by cycle;
  * the repaired Algorithm 1 satisfies all of that straight from its closed
    form, and reproduces the published control law of lines 7-11;
  * the datapath computes exact FFTs, IFFTs and butterfly-linear layers;
  * Algorithm 3 is a correct bit reversal, conflict free on both memories, in
    exactly N/P cycles;
  * measured cycles equal Eq. (10), including the abstract's 15,360;
  * the descriptor program is well formed and the sequencer never beats the
    datapath it drives.
"""

import math
import unittest

import numpy as np

import bspnet as bs
import flexbe as fx
import microcode as mc


# ---------------------------------------------------------------------------
class TestBitHelpers(unittest.TestCase):

    def test_bit_rev(self):
        self.assertEqual(fx.bit_rev(1, 3), 4)
        self.assertEqual(fx.bit_rev(0b1011, 4), 0b1101)
        for n in (1, 3, 5, 8):
            v = fx.bit_rev_array(n)
            self.assertTrue(np.array_equal(v[v], np.arange(1 << n)))

    def test_insert_zero(self):
        self.assertEqual(fx.insert_zero(0b111, 0), 0b1110)
        self.assertEqual(fx.insert_zero(0b111, 1), 0b1101)
        self.assertEqual(fx.insert_zero(0b111, 3), 0b0111)

    def test_bsm_scalar_matches_vector(self):
        for m in (1, 2, 4, 5):
            idx = np.arange(1 << (m + 3))
            ref = np.array([fx.bsm(int(i), m) for i in idx])
            self.assertTrue(np.array_equal(ref, fx.bsm_array(idx, m)))

    def test_shift_down_is_a_bijection(self):
        for n, m in [(3, 2), (5, 2), (8, 4), (10, 5)]:
            idx = np.arange(1 << n)
            slots = fx.bsm_array(idx, m) * (1 << (n - m)) + (idx >> m)
            self.assertEqual(len(np.unique(slots)), 1 << n)

    def test_row_is_a_rotation_of_the_banks(self):
        """A RAM row occupies every bank once, rotated by popcount(H)."""
        m = 3
        for H in range(8):
            row = np.arange(H << m, (H << m) + (1 << m))
            banks = fx.bsm_array(row, m)
            self.assertEqual(sorted(banks.tolist()), list(range(1 << m)))
            self.assertEqual(banks[0], fx.popcount(H) % (1 << m))


# ---------------------------------------------------------------------------
class TestAlgorithm1(unittest.TestCase):
    """The repaired listing, verified from its closed form."""

    CASES = [(3, 1), (4, 2), (6, 2), (6, 4), (9, 4), (10, 16), (12, 16), (5, 16)]

    def _cases(self):
        for n, P_bu in self.CASES:
            if n >= (2 * P_bu).bit_length() - 1:
                yield n, P_bu

    def test_conflict_free_and_complete(self):
        for n, P_bu in self._cases():
            m = (2 * P_bu).bit_length() - 1
            N, P = 1 << n, 1 << m
            with self.subTest(N=N, P_bu=P_bu):
                for k in range(n):
                    h, seen = n - 1 - k, np.zeros(N, dtype=bool)
                    for j in range(N // P):
                        c = fx.algorithm1(n, P_bu, k, j)
                        banks = fx.bsm_array(c.indices, m)
                        self.assertEqual(len(np.unique(banks)), P, "conflict")
                        self.assertFalse(seen[c.indices].any(), "reuse")
                        seen[c.indices] = True
                        self.assertTrue(np.all(c.indices[1::2] - c.indices[0::2]
                                               == (1 << h)))
                        self.assertFalse(np.any((c.indices[0::2] >> h) & 1))
                    self.assertTrue(seen.all(), "incomplete coverage")

    def test_published_lines_7_to_11(self):
        """R = bsm(I[0]); S = n-k-1 on (n-m)..(n-2), else 0."""
        for n, P_bu in self._cases():
            m = (2 * P_bu).bit_length() - 1
            for k in range(n):
                want = (n - k - 1) if (n - m) <= k <= (n - 2) else 0
                for j in range(1 << (n - m)):
                    c = fx.algorithm1(n, P_bu, k, j)
                    self.assertEqual(c.R, fx.bsm(int(c.indices[0]), m))
                    self.assertEqual(c.S, want)

    def test_prs_replaces_the_crossbar(self):
        """P_f = P_s x P_r for every cycle the formula emits."""
        n, P_bu = 10, 16
        m = (2 * P_bu).bit_length() - 1
        prs = fx.PermuteRotateSwitch(m)
        rng = np.random.default_rng(0)
        for k in range(n):
            for j in range(1 << (n - m)):
                c = fx.algorithm1(n, P_bu, k, j)
                banks = fx.bsm_array(c.indices, m)
                self.assertTrue(np.array_equal(prs.read_map(c.R, c.S), banks))
                D = rng.normal(size=1 << m) + 1j * rng.normal(size=1 << m)
                Pf = fx.fcs_matrix(c.indices, m)
                self.assertTrue(np.allclose(prs.forward(D, c.R, c.S),
                                            D[np.argmax(Pf, axis=1)]))
                X = prs.forward(D, c.R, c.S)
                self.assertTrue(np.allclose(prs.inverse(X, c.R, c.S), D))

    def test_subset_switch_uses_at_most_m_states(self):
        sched = fx.ButterflySchedule(12, 16)
        states = {c.S for c in sched.all_cycles()}
        self.assertLessEqual(len(states), sched.m)
        self.assertTrue(states <= set(range(sched.m)))

    def test_schedule_validates(self):
        for n, P_bu in self._cases():
            with self.subTest(N=1 << n, P_bu=P_bu):
                s = fx.ButterflySchedule(n, P_bu, validate=True)
                self.assertEqual(s.total_cycles, (1 << n) // (2 * P_bu) * n)

    def test_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            fx.algorithm1(10, 16, 10, 0)
        with self.assertRaises(ValueError):
            fx.algorithm1(10, 16, 0, 32)


# ---------------------------------------------------------------------------
class TestTransforms(unittest.TestCase):

    def test_fft_matches_numpy(self):
        for N, P_bu in [(32, 2), (64, 4), (256, 4), (1024, 16), (4096, 16)]:
            with self.subTest(N=N, P_bu=P_bu):
                rng = np.random.default_rng(N + P_bu)
                x = rng.normal(size=N) + 1j * rng.normal(size=N)
                eng = fx.FlexBE(P_bu)
                y, _ = eng.transform(x, bitrev=True, datapath="cycle",
                                     validate=True)
                ref = np.fft.fft(x)
                self.assertLess(np.max(np.abs(y[0] - ref)) / np.max(np.abs(ref)),
                                1e-12)

    def test_cycle_and_vector_datapaths_agree(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=256) + 1j * rng.normal(size=256)
        eng = fx.FlexBE(4)
        a, _ = eng.transform(x, bitrev=True, datapath="cycle")
        b, _ = eng.transform(x, bitrev=True, datapath="vector")
        self.assertTrue(np.allclose(a, b))

    def test_output_is_bit_reversed_without_writeback(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=256) + 1j * rng.normal(size=256)
        y, _ = fx.FlexBE(4).transform(x, bitrev=False, datapath="cycle")
        self.assertTrue(np.allclose(y[0][fx.bit_rev_array(8)], np.fft.fft(x)))

    def test_packing_multiple_sequences(self):
        N, P_bu, P_N = 128, 4, 5
        rng = np.random.default_rng(7)
        x = rng.normal(size=(P_N, N)) + 1j * rng.normal(size=(P_N, N))
        y, st = fx.FlexBE(P_bu).transform(x, bitrev=True, datapath="cycle")
        self.assertTrue(np.allclose(y, np.fft.fft(x, axis=1)))
        self.assertEqual(st.ram_depth, P_N * N // (2 * P_bu))
        self.assertEqual(st.butterfly_cycles,
                         P_N * N // (2 * P_bu) * int(math.log2(N)))

    def test_inverse_transform(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=64) + 1j * rng.normal(size=64)
        eng = fx.FlexBE(4)
        X, _ = eng.transform(x, bitrev=True, datapath="cycle")
        y, _ = eng.transform(X[0], fx.fft_coefficients(64, inverse=True),
                             bitrev=True, datapath="cycle")
        self.assertTrue(np.allclose(y[0] / 64, x))

    def test_bl_layer_equals_its_dense_matrix(self):
        rng = np.random.default_rng(2)
        C = fx.random_bl_coefficients(32, rng)
        eng = fx.FlexBE(16)
        M = np.asarray(eng.transform(np.eye(32), C, datapath="cycle")[0]).T
        x = rng.normal(size=(7, 32))
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
                _, P_sub = eng.geometry(l)
                comp = eng.interleave(x.astype(complex), P_sub)
                ref = fx.butterfly_reference(comp,
                                             fx.expand_coefficients(C, P_sub),
                                             P_sub)
                self.assertTrue(np.allclose(
                    y, eng.deinterleave(ref, l, P_sub, 4)))

    def test_paper_weight_layout(self):
        l = 16
        W = np.random.default_rng(1).normal(size=(2, 2, l // 2, 4))
        C = fx.coefficients_from_paper_layout(W)
        self.assertEqual(C.shape, (4, l // 2, 2, 2))
        self.assertEqual(C[3, 5, 1, 0], W[1, 0, 5, 3])


# ---------------------------------------------------------------------------
class TestSubParallel(unittest.TestCase):

    def test_geometry(self):
        eng = fx.FlexBE(16)
        self.assertEqual(eng.geometry(8), (32, 4))
        self.assertEqual(eng.geometry(32), (32, 1))
        self.assertEqual(eng.geometry(1024), (1024, 1))

    def test_short_transforms(self):
        for P_bu, l, n_seq in [(4, 4, 6), (4, 2, 8), (16, 8, 13), (16, 4, 4)]:
            with self.subTest(P_bu=P_bu, l=l):
                rng = np.random.default_rng(l * 31 + P_bu)
                x = rng.normal(size=(n_seq, l)) + 1j * rng.normal(size=(n_seq, l))
                y, st = fx.FlexBE(P_bu).transform(x, bitrev=True,
                                                  datapath="cycle")
                self.assertEqual(st.P_sub, 2 * P_bu // l)
                self.assertEqual(st.n_stages, int(math.log2(l)))
                self.assertTrue(np.allclose(y, np.fft.fft(x, axis=1)))

    def test_interleaving_is_eq5(self):
        eng = fx.FlexBE(4)
        x = np.array([[0, 1, 2, 3], [10, 11, 12, 13]])
        comp = eng.interleave(x, 2)
        self.assertTrue(np.array_equal(comp[0], [0, 10, 1, 11, 2, 12, 3, 13]))
        self.assertTrue(np.array_equal(eng.deinterleave(comp, 4, 2, 2), x))

    def test_beats_zero_padding(self):
        flex = fx.layer_cycles(4096, 8, P_be=4, P_bu=16)
        pad = fx.layer_cycles(4096, 32, P_be=4, P_bu=16)
        self.assertAlmostEqual(pad / flex, 4 * (5 / 3), places=6)


# ---------------------------------------------------------------------------
class TestBitReversal(unittest.TestCase):

    def test_algorithm3_is_a_conflict_free_permutation(self):
        for n, m in [(4, 2), (6, 2), (10, 5), (12, 5), (15, 5), (8, 4)]:
            with self.subTest(N=1 << n, P=1 << m):
                j, k = fx.bitrev_schedule(n, m)
                P, N = 1 << m, 1 << n
                self.assertEqual(j.shape, (N // P, P))
                self.assertEqual(len(np.unique(j)), N)
                self.assertTrue(np.array_equal(k, fx.bit_rev_array(n)[j]))
                for jv, kv in zip(j, k):
                    self.assertTrue(np.array_equal(fx.bsm_array(jv, m),
                                                   np.arange(P)))
                    self.assertEqual(len(np.unique(kv & (P - 1))), P)

    def test_eq7_destination_bank_law(self):
        n, m = 12, 5
        P = 1 << m
        j, k = fx.bitrev_schedule(n, m)
        brev = fx.bit_rev_array(m)
        cyc = 0
        for u in range(P):
            expect = brev[(np.arange(P) + u) % P]
            for _ in range(j.shape[0] // P):
                self.assertTrue(np.array_equal(k[cyc] & (P - 1), expect))
                cyc += 1

    def test_cycle_count(self):
        _, st = fx.FlexBE(16).transform(
            np.random.default_rng(5).normal(size=1024) + 0j,
            bitrev=True, datapath="cycle")
        self.assertEqual(st.bitrev_cycles, 1024 // 32)

    def test_short_transform_falls_back_cleanly(self):
        """N < P^2 is outside Algorithm 3's regime: model it, do not claim
        conflict freedom, and still produce the right permutation."""
        for N, P_bu in [(256, 16), (512, 16), (32, 16)]:
            with self.subTest(N=N, P_bu=P_bu):
                self.assertLess(N, (2 * P_bu) ** 2)
                rng = np.random.default_rng(N)
                x = rng.normal(size=N) + 1j * rng.normal(size=N)
                y, st = fx.FlexBE(P_bu).transform(x, bitrev=True,
                                                  datapath="cycle")
                self.assertTrue(np.allclose(y[0], np.fft.fft(x)))
                self.assertEqual(st.bitrev_cycles, max(1, N // (2 * P_bu)))
        with self.assertRaises(ValueError):
            fx.bitrev_schedule(8, 5)


# ---------------------------------------------------------------------------
class TestFixedPoint(unittest.TestCase):

    def test_quantiser(self):
        fmt = fx.FixedPointFormat()
        self.assertAlmostEqual(fmt.max_val, 1.0 - 2 ** -15, places=9)
        self.assertEqual(fmt.min_val, -1.0)
        self.assertEqual(fmt.quantise(2.0), fmt.max_val)
        self.assertEqual(fmt.quantise(-5.0), fmt.min_val)
        self.assertAlmostEqual(float(fmt.quantise(0.5 + 2 ** -16)), 0.5, places=9)

    def test_fxp16_fft_accuracy(self):
        fmt = fx.FixedPointFormat()
        rng = np.random.default_rng(9)
        x = fmt.quantise((rng.normal(size=1024) + 1j * rng.normal(size=1024)) / 8)
        y, st = fx.FlexBE(16, fmt=fmt).transform(x, bitrev=True,
                                                 stage_scale=0.5,
                                                 datapath="cycle")
        ref = np.fft.fft(x) * st.scale
        err = np.linalg.norm(y[0] - ref) / np.linalg.norm(ref)
        self.assertLess(err, 1e-2)
        self.assertGreater(20 * math.log10(1 / err), 40.0)


# ---------------------------------------------------------------------------
class TestPerformanceModel(unittest.TestCase):

    def test_measured_cycles_match_eq10(self):
        for N, P_bu, P_N in [(64, 4, 3), (256, 16, 2), (32, 16, 8), (8, 16, 16)]:
            with self.subTest(N=N, P_bu=P_bu):
                rng = np.random.default_rng(N)
                _, st = fx.FlexBE(P_bu).transform(rng.normal(size=(P_N, N)),
                                                  datapath="cycle")
                self.assertAlmostEqual(
                    st.butterfly_cycles,
                    fx.cycles_eq10(P_N, st.N, st.P_sub, 1, P_bu), places=6)

    def test_abstract_headline_15360(self):
        self.assertEqual(fx.cycles_eq10(4, 32768, 1, P_be=4, P_bu=16), 15360)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 32768)) + 1j * rng.normal(size=(4, 32768))
        y, st = fx.BEArray(4, 16).transform(x, bitrev=False, datapath="vector")
        self.assertEqual(st.butterfly_cycles, 15360)
        self.assertTrue(np.allclose(y, np.fft.fft(x, axis=1)[:,
                                                             fx.bit_rev_array(15)]))

    def test_prs_cheaper_than_fcs(self):
        prev = 0.0
        for P_bu in (4, 8, 16, 32, 64):
            c = fx.interconnect_cost(P_bu)
            self.assertLess(c["prs_mux2"], c["fcs_mux2"])
            self.assertGreater(c["ratio"], prev)
            prev = c["ratio"]
        self.assertGreaterEqual(fx.interconnect_cost(2)["prs_mux2"],
                                fx.interconnect_cost(2)["fcs_mux2"])
        c8, c64 = fx.interconnect_cost(8), fx.interconnect_cost(64)
        self.assertGreater(c64["fcs_mux2"] / c8["fcs_mux2"], 60.0)
        self.assertLess(c64["prs_mux2"] / c8["prs_mux2"], 16.0)


# ---------------------------------------------------------------------------
class TestBSPNet(unittest.TestCase):

    def test_feature_extraction_matches_numpy(self):
        cfg = bs.BSPNetConfig("small", d_in=8, d_m=32, L=32, n_mlp_blocks=2,
                              bit_reversal=True, N=1024)
        net = bs.BSPNet(cfg, bs.HW["FlexBE"], seed=0)
        rng = np.random.default_rng(4)
        s = (rng.normal(size=cfg.N) + 1j * rng.normal(size=cfg.N)) / 8
        got, ref = net.features(s), net.features_reference(s)
        for key in ref:
            self.assertTrue(np.allclose(got[key], ref[key], atol=1e-8), key)

    def test_forward_is_deterministic(self):
        cfg = bs.BSPNetConfig("small", d_in=8, d_m=32, L=64, n_mlp_blocks=2,
                              bit_reversal=True, N=1024)
        net = bs.BSPNet(cfg, bs.HW["FlexBE"], seed=1)
        rng = np.random.default_rng(6)
        s = (rng.normal(size=cfg.N) + 1j * rng.normal(size=cfg.N)) / 8
        a = net.forward(s)
        self.assertEqual(a.shape, (cfg.n_classes,))
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.allclose(a, net.forward(s)))

    def test_cfg6_latency(self):
        cfg = bs.CONFIGS["cfg-6"]
        lat = bs.latency_ms(cfg, bs.HW["BSP-Flex"])
        self.assertGreater(lat, 0.05)
        self.assertLess(lat, 0.5)
        self.assertGreater(bs.throughput_sps(cfg, bs.HW["BSP-Flex"]), 2000)

    def test_flexbe_beats_be_base_everywhere(self):
        for name, cfg in bs.CONFIGS.items():
            with self.subTest(cfg=name):
                flex = bs.latency_ms(cfg, bs.HW["FlexBE"])
                for base in ("BE-base-1", "BE-base-2"):
                    self.assertLess(flex, bs.latency_ms(cfg, bs.HW[base]))

    def test_fft_layer_is_the_headline_number(self):
        items = dict(bs.cycle_breakdown(bs.CONFIGS["cfg-6"], bs.HW["FlexBE"]))
        self.assertEqual([v for k, v in items.items() if "FFT" in k][0], 15360)


# ---------------------------------------------------------------------------
class TestMicrocode(unittest.TestCase):

    def setUp(self):
        self.cfg, self.hw = bs.CONFIGS["cfg-6"], bs.HW["BSP-Flex"]

    def test_descriptor_round_trip(self):
        d = mc.Descriptor("bs.bfly", mode=mc.MODES["BL"], log2_l=5,
                          log2_Psub=2, P_N=1024, src=0x1000, dst=0x2000,
                          coeff=0x30, count=32768, tag=7,
                          flags=mc.FLAGS["RELU"])
        raw = d.encode()
        self.assertEqual(len(raw), mc.Descriptor.SIZE)
        self.assertEqual(mc.Descriptor.decode(raw), d)

    def test_every_opcode_encodes(self):
        for op in mc.OPCODES:
            d = mc.Descriptor(op)
            self.assertEqual(mc.Descriptor.decode(d.encode()).opcode, op)
            self.assertIn(d.resource, ("dma_in", "dma_out", "compute"))

    def test_program_is_well_formed(self):
        cmds = mc.build_program(self.cfg, self.hw)
        self.assertGreater(len(cmds), 50)
        self.assertLess(len(cmds), 200)
        for n, c in enumerate(cmds):
            self.assertTrue(all(d < n for d in c.deps))
            self.assertEqual(c.desc.tag, n)
        self.assertEqual(cmds[0].kind, "dma_in")
        self.assertEqual(cmds[-1].kind, "dma_out")

    def test_program_carries_the_bspnet_cycles(self):
        cmds = mc.build_program(self.cfg, self.hw)
        cyc = sum(c.cycles for c in cmds if c.kind == "compute")
        base = bs.total_cycles(self.cfg, self.hw)
        self.assertGreaterEqual(cyc, base)          # plus pow / mag / pool glue
        self.assertLess(cyc / base, 1.15)

    def test_latency_never_beats_the_datapath(self):
        for key, p in mc.PLATFORMS.items():
            with self.subTest(platform=key):
                rep = mc.Sequencer(p).run(
                    mc._stream(self.cfg, self.hw, 1, p.input_chunks))
                self.assertGreaterEqual(rep.latency_us, rep.compute_us)

    def test_pynq_is_host_bound_and_rings_are_not(self):
        rows = {r["key"]: r for r in mc.compare_platforms()}
        self.assertEqual(rows["pynq"]["critical"], "host")
        self.assertGreater(rows["pynq"]["overhead_pct"], 100.0)
        for key in ("ring", "static", "tuned"):
            self.assertEqual(rows[key]["critical"], "compute")
            self.assertLess(rows[key]["overhead_pct"], 15.0)

    def test_ring_beats_mmio(self):
        rows = {r["key"]: r for r in mc.compare_platforms()}
        self.assertLess(rows["static"]["latency_us"], rows["mmio"]["latency_us"])

    def test_prefetch_depth_saturates(self):
        lat = [r["latency_us"] for r in mc.ring_depth_sweep()]
        self.assertEqual(lat, sorted(lat, reverse=True))
        self.assertAlmostEqual(lat[-1], lat[-2], places=6)

    def test_per_resource_queues_help_across_inferences(self):
        shared, split = mc.PLATFORMS["static"], mc.PLATFORMS["tuned"]
        many = [mc.Sequencer(p).run(mc._stream(self.cfg, self.hw, 8,
                                               p.input_chunks))
                for p in (shared, split)]
        self.assertLess(many[1].per_inference_us, many[0].per_inference_us)

    def test_batch_amortises_towards_peak(self):
        rows = mc.batch_sweep(batches=(1, 2, 4, 8, 10))
        tuned = [r["tuned"] for r in rows]
        self.assertEqual(tuned, sorted(tuned))
        self.assertGreater(tuned[-1] / rows[-1]["peak"], 0.85)
        self.assertLess(rows[0]["pynq"] / rows[0]["peak"], 0.4)

    def test_double_buffering_matters(self):
        p = mc.PLATFORMS["static"]
        off = mc.replace(p, double_buffer=False)
        cmds = mc._stream(self.cfg, self.hw, 4, 1)
        self.assertLess(mc.Sequencer(p).run(cmds).latency_us,
                        mc.Sequencer(off).run(cmds).latency_us)

    def test_link_is_not_the_constraint(self):
        b = mc.port_budget()
        self.assertLess(b["utilisation"], 0.25)
        self.assertEqual(b["min_hp_ports"], 1)

    def test_sequencer_area_is_small_and_fits(self):
        a = mc.sequencer_area()
        self.assertLess(a["lut_overhead_pct"], 5.0)
        self.assertTrue(a["fits"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
