// =============================================================================
// CycleControl.scala -- per-cycle index vectors, addresses and PRS control.
//
// Combinational realisation of cycle_control(n, P_bu, k, j) from flexbe.py.
// The engine runs decimation-in-frequency, so stage k pairs indices differing
// in the hole bit h = n-1-k.  Conflict freedom under bsm forces two regimes:
//
//   h < m  : both operands share one RAM row H = j
//              I[2t]   = (H<<m) + ins(t,h)        S = h
//   h >= m : operands span rows H and H+2^(h-m)
//              e = j&1,  H = ins(j>>1, h-m)
//              I[2t]   = (H<<m) + 2t + e          S = 0
//   both   : I[2t+1] = I[2t] + 2^h,   R = bsm(I[0])
//
// Outputs, all for the current (k, j):
//   idx(2*P_bu)   logical element indices in butterfly-slot order
//   addr(P)       per-bank read/write depth (a_x); addr(bsm(idx(s))) = idx(s)>>m
//   R, S          PRS rotation and subset state
//   coeff(P_bu)   coefficient (twiddle / weight) address per butterfly
//
// Parameters: N = 2^nMax is the largest transform; P_bu fixes m = log2(2*P_bu).
// The instantaneous transform length is nCur (<= nMax), supplied at run time.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

class CycleControl(val nMax: Int, val pBu: Int) extends Module {
  val P    = 2 * pBu
  val m    = log2Ceil(P)
  val idxW = nMax
  val sW   = if (m > 1) log2Ceil(m) else 1

  val io = IO(new Bundle {
    val nCur = Input(UInt(log2Ceil(nMax + 1).W))   // stages = nCur (for P_sub=1)
    val k    = Input(UInt(log2Ceil(nMax).W))       // stage index
    val j    = Input(UInt((nMax - m + 1).W))       // cycle index within stage
    val idx  = Output(Vec(P, UInt(idxW.W)))
    val addr = Output(Vec(P, UInt((nMax - m).W)))
    val R    = Output(UInt(m.W))
    val S    = Output(UInt(sW.W))
    val coeff = Output(Vec(pBu, UInt((nMax - 1).W)))
  })

  val h = io.nCur - 1.U - io.k                      // hole bit
  val inRow = h < m.U                               // regime selector

  // --- lower indices I[2t] for t = 0..P_bu-1 -------------------------------
  val lower = Wire(Vec(pBu, UInt(idxW.W)))
  // regime h < m : H = j, I[2t] = (H<<m) + ins(t,h)
  //   ins(t,h) needs h in [0, m-1]; t in [0, P_bu-1] (m-1 bits)
  // regime h >= m: e = j&1, H = ins(j>>1, h-m), I[2t] = (H<<m) + 2t + e
  val e   = io.j(0)
  val jHi = (io.j >> 1).asUInt
  for (t <- 0 until pBu) {
    // in-row branch
    val insT = FlexBits.insertZero(t.U((if (m>1) m-1 else 1).W), h, m - 1, m)
    val hIn  = (io.j << m.U).asUInt | insT
    // two-row branch: H = ins(jHi, h-m); low part of I[2t] = 2t + e
    val hmm  = h - m.U
    val Htwo = FlexBits.insertZero(jHi, hmm, nMax - m, nMax - m + 1)
    val hTwo = (Htwo << m.U).asUInt | (2 * t).U | e
    lower(t) := Mux(inRow, hIn, hTwo)(idxW - 1, 0)
  }

  // --- full index vector, interleaved lower/upper with stride 2^h ----------
  val twoH = (1.U << h).asUInt
  for (t <- 0 until pBu) {
    io.idx(2 * t)     := lower(t)
    io.idx(2 * t + 1) := (lower(t) + twoH)(idxW - 1, 0)
  }

  // --- addresses: addr(bank) = idx >> m, scattered by bsm ------------------
  val addrInit = WireDefault(VecInit(Seq.fill(P)(0.U((nMax - m).W))))
  for (s <- 0 until P) {
    val b = FlexBits.bsm(io.idx(s), m)
    addrInit(b) := (io.idx(s) >> m.U).asUInt
  }
  io.addr := addrInit

  // --- PRS control ---------------------------------------------------------
  io.R := FlexBits.bsm(io.idx(0), m)
  io.S := Mux(inRow, h(sW - 1, 0), 0.U)

  // --- coefficient address per butterfly -----------------------------------
  //   a = I[2t];  coeff = ((a >> (h+1)) << h) | (a & (2^h - 1))
  for (t <- 0 until pBu) {
    val a   = lower(t)
    val hi  = (a >> (h + 1.U)).asUInt
    val loM = (1.U << h).asUInt - 1.U
    io.coeff(t) := ((hi << h).asUInt | (a & loM))(nMax - 2, 0)
  }
}
