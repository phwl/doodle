// =============================================================================
// Engine.scala -- one butterfly engine: the per-cycle loop of Fig. 2.
//
// Pipeline (BSPM is a SyncReadMem, so read data is valid the cycle after the
// address):
//   cycle t   : control(k,j) presents read addresses  -> BSPM read issued
//   cycle t+1 : rData(t) valid -> PRS-A -> butterflies -> PRS-B -> write back,
//               using control(k,j) held one cycle in a delay register.
// Consecutive cycles touch disjoint (bank, depth) slots, so a one-deep
// read/compute/write pipeline has no read-after-write hazard between adjacent
// cycles; a final drain cycle commits the last write.
//
// Preload / read-back reuse the engine's own single read/write ports while the
// FSM is idle, driven by the debug interface below.  A testbench can therefore
// load BSPM in logical-index space (the hardware applies the bsm skew), run one
// transform, and read the result back, so the datapath is verified numerically
// against a reference FFT without adding a second memory port.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

class EngineConfig(
  val nMax: Int = 15,          // largest transform: 2^15 = 32768
  val pBu: Int = 16,           // butterfly units -> P = 32 banks
  val depth: Int = 8192,       // BSPM words per bank (P_N packing)
  val coeffDepth: Int = 4096,  // CRAM entries
  val intBits: Int = 1,
  val fracBits: Int = 15
)

class Engine(cfg: EngineConfig = new EngineConfig) extends Module {
  val P = 2 * cfg.pBu
  val m = log2Ceil(P)
  val depthW = log2Ceil(cfg.depth)
  val sW = if (m > 1) log2Ceil(m) else 1

  val io = IO(new Bundle {
    val start     = Input(Bool())
    val nCur      = Input(UInt(log2Ceil(cfg.nMax + 1).W))
    val scale     = Input(Bool())
    val coeffBase = Input(UInt(log2Ceil(cfg.coeffDepth).W))
    val busy      = Output(Bool())
    val done      = Output(Bool())

    // ---- debug interface, active only while idle (reuses engine ports) ----
    val dbgWen  = Input(Bool())                              // BSPM write
    val dbgWidx = Input(UInt(cfg.nMax.W))                    // logical index
    val dbgWval = Input(new Cplx(cfg.intBits, cfg.fracBits))
    val dbgRidx = Input(UInt(cfg.nMax.W))                    // logical index
    val dbgRval = Output(new Cplx(cfg.intBits, cfg.fracBits))
    val cWen    = Input(Bool())                              // CRAM write
    val cWaddr  = Input(UInt(log2Ceil(cfg.coeffDepth).W))
    val cWval   = Input(Vec(4, new Cplx(cfg.intBits, cfg.fracBits)))

    val dbgR    = Output(UInt(m.W))
    val dbgS    = Output(UInt(sW.W))
  })

  val bspm = Module(new BankedMemory(P, cfg.depth, cfg.intBits, cfg.fracBits))
  val ctl  = Module(new CycleControl(cfg.nMax, cfg.pBu))
  val prsA = Module(new PRS(P))
  val prsB = Module(new PRSInverse(P))
  val bus  = Seq.fill(cfg.pBu)(Module(new ButterflyUnit(cfg.intBits, cfg.fracBits)))
  val cram = SyncReadMem(cfg.coeffDepth, Vec(4, new Cplx(cfg.intBits, cfg.fracBits)))

  when(io.cWen) { cram.write(io.cWaddr, io.cWval) }

  // ---- control FSM --------------------------------------------------------
  val sIdle :: sRun :: sDrain :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val k = RegInit(0.U(log2Ceil(cfg.nMax).W))
  val j = RegInit(0.U((cfg.nMax - m + 1).W))
  val cyclesPerStage = (1.U << (io.nCur - m.U)).asUInt
  val lastJ = cyclesPerStage - 1.U
  val lastK = io.nCur - 1.U
  val running = state === sRun

  io.busy := state =/= sIdle
  io.done := false.B
  when(state === sIdle) {
    when(io.start) { state := sRun; k := 0.U; j := 0.U }
  }.elsewhen(state === sRun) {
    when(j === lastJ) {
      j := 0.U
      when(k === lastK) { state := sDrain }.otherwise { k := k + 1.U }
    }.otherwise { j := j + 1.U }
  }.elsewhen(state === sDrain) {
    state := sIdle
    io.done := true.B
  }

  ctl.io.nCur := io.nCur
  ctl.io.k := k
  ctl.io.j := j

  // ---- BSPM read port: engine addresses while running, debug idx when idle
  val dbgRbank  = FlexBits.bsm(io.dbgRidx, m)
  val dbgRdepth = (io.dbgRidx >> m.U).asUInt
  for (b <- 0 until P) {
    bspm.io.rAddr(b) := Mux(running, ctl.io.addr(b),
                           Mux(b.U === dbgRbank, dbgRdepth, 0.U))
  }
  io.dbgRval := bspm.io.rData(RegNext(dbgRbank))   // 1-cycle SyncReadMem latency

  // ---- pipeline registers: hold control one cycle to meet rData latency ---
  val addrD  = RegNext(ctl.io.addr)
  val rD     = RegNext(ctl.io.R)
  val sD     = RegNext(ctl.io.S)
  val coeffD = RegNext(ctl.io.coeff)
  val scaleD = RegNext(io.scale)
  val wEnD   = RegNext(running)

  // ---- PRS-A gather -------------------------------------------------------
  prsA.io.R := rD
  prsA.io.S := sD
  for (b <- 0 until P) prsA.io.din(b) := bspm.io.rData(b)

  // ---- butterflies --------------------------------------------------------
  // Coefficients are stage-major: CRAM address = coeffBase + k*(l/2) + coeff,
  // with l/2 = 2^(nCur-1).  The stage base is registered alongside the other
  // pipelined control so it aligns with coeffD.
  val kD        = RegNext(k)
  val nCurD     = RegNext(io.nCur)
  val stageBase = (kD << (nCurD - 1.U)).asUInt
  for (t <- 0 until cfg.pBu) {
    val cw = cram.read((coeffD(t) + stageBase + io.coeffBase)(log2Ceil(cfg.coeffDepth) - 1, 0))
    bus(t).io.a := prsA.io.dout(2 * t)
    bus(t).io.b := prsA.io.dout(2 * t + 1)
    bus(t).io.c00 := cw(0); bus(t).io.c01 := cw(1)
    bus(t).io.c10 := cw(2); bus(t).io.c11 := cw(3)
    bus(t).io.scale := scaleD
  }

  // ---- PRS-B scatter ------------------------------------------------------
  prsB.io.R := rD
  prsB.io.S := sD
  for (t <- 0 until cfg.pBu) {
    prsB.io.xin(2 * t)     := bus(t).io.out0
    prsB.io.xin(2 * t + 1) := bus(t).io.out1
  }

  // ---- BSPM write port: pipelined engine write, or debug write when idle --
  // Per-bank write strobes let a debug write commit exactly one bank while the
  // engine write drives all banks; the two never coincide (debug only idle).
  val dbgWbank  = FlexBits.bsm(io.dbgWidx, m)
  val dbgWdepth = (io.dbgWidx >> m.U).asUInt
  for (b <- 0 until P) {
    val engineW = wEnD
    val debugW  = io.dbgWen && !running && (b.U === dbgWbank)
    bspm.io.wEn(b)   := engineW || debugW
    bspm.io.wAddr(b) := Mux(engineW, addrD(b), dbgWdepth)
    bspm.io.wData(b) := Mux(engineW, prsB.io.dout(b), io.dbgWval)
  }

  io.dbgR := ctl.io.R
  io.dbgS := ctl.io.S
}
