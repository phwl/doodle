// =============================================================================
// Engine.scala -- one butterfly engine: the per-cycle loop of Fig. 2.
//
// Each cycle:
//   1. CycleControl emits idx / addr / R / S / coeff for (k, j)
//   2. read one word per bank from BSPM at addr
//   3. PRS-A gathers bank order -> butterfly-slot order
//   4. P_bu butterfly units apply the 2x2 coefficients from CRAM
//   5. PRS-B scatters slot order -> bank order
//   6. write back in place to BSPM at addr
//
// A simple FSM drives (k, j) over stages*cycles_per_stage for one bs.bfly
// command.  BSPM is a SyncReadMem, so read data is available the cycle after
// the address; the loop is pipelined one deep and the write of cycle t occurs
// with the control of cycle t (addresses are identical for read and write in
// this in-place datapath).
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

  val io = IO(new Bundle {
    val start   = Input(Bool())
    val nCur    = Input(UInt(log2Ceil(cfg.nMax + 1).W))  // stages for this cmd
    val scale   = Input(Bool())                          // FFT stage scaling
    val coeffBase = Input(UInt(log2Ceil(cfg.coeffDepth).W))
    val busy    = Output(Bool())
    val done    = Output(Bool())
    // verification hooks
    val dbgIdx  = Output(Vec(P, UInt(cfg.nMax.W)))
    val dbgR    = Output(UInt(m.W))
    val dbgS    = Output(UInt((if (m>1) log2Ceil(m) else 1).W))
  })

  val bspm = Module(new BankedMemory(P, cfg.depth, cfg.intBits, cfg.fracBits))
  val ctl  = Module(new CycleControl(cfg.nMax, cfg.pBu))
  val prsA = Module(new PRS(P))
  val prsB = Module(new PRSInverse(P))
  val bus  = Seq.fill(cfg.pBu)(Module(new ButterflyUnit(cfg.intBits, cfg.fracBits)))

  // coefficient RAM: P_bu*2*2 words addressed by coeff index (packed 2x2)
  val cram = SyncReadMem(cfg.coeffDepth, Vec(4, new Cplx(cfg.intBits, cfg.fracBits)))

  // ---- control FSM: iterate (k, j) ----------------------------------------
  val sIdle :: sRun :: sDrain :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val k = RegInit(0.U(log2Ceil(cfg.nMax).W))
  val j = RegInit(0.U((cfg.nMax - m + 1).W))
  val cyclesPerStage = (1.U << (io.nCur - m.U)).asUInt   // N/P for P_sub=1
  val lastJ = cyclesPerStage - 1.U
  val lastK = io.nCur - 1.U

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

  // ---- read ---------------------------------------------------------------
  for (b <- 0 until P) bspm.io.rAddr(b) := ctl.io.addr(b)

  // ---- PRS-A gather -------------------------------------------------------
  prsA.io.R := ctl.io.R
  prsA.io.S := ctl.io.S
  for (b <- 0 until P) prsA.io.din(b) := bspm.io.rData(b)

  // ---- butterflies --------------------------------------------------------
  val coeffRd = cram.read(ctl.io.coeff(0) + io.coeffBase)  // one shared example
  for (t <- 0 until cfg.pBu) {
    val cw = cram.read(ctl.io.coeff(t) + io.coeffBase)     // 2x2 = [c00,c01,c10,c11]
    bus(t).io.a := prsA.io.dout(2 * t)
    bus(t).io.b := prsA.io.dout(2 * t + 1)
    bus(t).io.c00 := cw(0); bus(t).io.c01 := cw(1)
    bus(t).io.c10 := cw(2); bus(t).io.c11 := cw(3)
    bus(t).io.scale := io.scale
  }

  // ---- PRS-B scatter ------------------------------------------------------
  prsB.io.R := ctl.io.R
  prsB.io.S := ctl.io.S
  for (t <- 0 until cfg.pBu) {
    prsB.io.xin(2 * t)     := bus(t).io.out0
    prsB.io.xin(2 * t + 1) := bus(t).io.out1
  }

  // ---- write back in place ------------------------------------------------
  bspm.io.wEn := state === sRun
  for (b <- 0 until P) {
    bspm.io.wAddr(b) := ctl.io.addr(b)
    bspm.io.wData(b) := prsB.io.dout(b)
  }

  io.dbgIdx := ctl.io.idx
  io.dbgR := ctl.io.R
  io.dbgS := ctl.io.S
}
