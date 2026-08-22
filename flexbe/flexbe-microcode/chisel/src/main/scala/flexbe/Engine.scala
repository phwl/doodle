// =============================================================================
// Engine.scala -- one butterfly engine: the per-cycle loop of Fig. 2.
//
// The banks are single-port, so a read and a write must not target one bank in
// the same cycle.  Each butterfly step therefore takes two phases:
//   phase R (issue): drive read addresses for (k,j); no write.
//   phase W (commit): rData valid -> PRS-A -> butterflies -> PRS-B -> write the
//                     results back to the SAME (k,j) addresses; no read issued.
// Adjacent steps do not overlap on the port, so behaviour is unambiguous on
// single-port SyncReadMem.  (A dual-port or true banked build could overlap the
// phases to reach one step per cycle; that is a later optimisation and does not
// change the functional result verified here.)
//
// Preload / read-back reuse the same ports while idle, via the debug interface.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

class EngineConfig(
  val nMax: Int = 15,
  val pBu: Int = 16,
  val depth: Int = 8192,
  val coeffDepth: Int = 4096,
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

    val dbgWen  = Input(Bool())
    val dbgWidx = Input(UInt(cfg.nMax.W))
    val dbgWval = Input(new Cplx(cfg.intBits, cfg.fracBits))
    val dbgRidx = Input(UInt(cfg.nMax.W))
    val dbgRval = Output(new Cplx(cfg.intBits, cfg.fracBits))
    val cWen    = Input(Bool())
    val cWaddr  = Input(UInt(log2Ceil(cfg.coeffDepth).W))
    val cWval   = Input(Vec(4, new Cplx(cfg.intBits, cfg.fracBits)))

    // stop after finishing this stage index (default: run all stages)
    val dbgMaxStage = Input(UInt(log2Ceil(cfg.nMax + 1).W))

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

  // ---- FSM: two phases per butterfly step ---------------------------------
  val sIdle :: sRead :: sWrite :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)
  val k = RegInit(0.U(log2Ceil(cfg.nMax).W))
  val j = RegInit(0.U((cfg.nMax - m + 1).W))
  val cyclesPerStage = (1.U << (io.nCur - m.U)).asUInt
  val lastJ = cyclesPerStage - 1.U
  val lastK = io.nCur - 1.U

  val running = (state === sRead) || (state === sWrite)
  io.busy := state =/= sIdle
  io.done := false.B

  switch(state) {
    is(sIdle)  { when(io.start) { state := sRead; k := 0.U; j := 0.U } }
    is(sRead)  { state := sWrite }                    // read issued this cycle
    is(sWrite) {                                       // commit; advance (k,j)
      when(j === lastJ) {
        j := 0.U
        when(k === lastK || k === io.dbgMaxStage) { state := sDone }
          .otherwise { k := k + 1.U; state := sRead }
      }.otherwise { j := j + 1.U; state := sRead }
    }
    is(sDone)  { state := sIdle; io.done := true.B }
  }

  ctl.io.nCur := io.nCur
  ctl.io.k := k
  ctl.io.j := j

  // ---- read port: drive addresses in sRead (engine) or when idle (debug) --
  val dbgRbank  = FlexBits.bsm(io.dbgRidx, m)
  val dbgRdepth = (io.dbgRidx >> m.U).asUInt
  for (b <- 0 until P) {
    val engRead = state === sRead
    bspm.io.rAddr(b) := Mux(engRead, ctl.io.addr(b),
                           Mux(b.U === dbgRbank, dbgRdepth, 0.U))
  }
  io.dbgRval := bspm.io.rData(RegNext(dbgRbank))

  // rData is valid in sWrite (the cycle after sRead).  Control (k,j) has not
  // advanced between sRead and sWrite, so ctl.io.* still describes this step.
  prsA.io.R := ctl.io.R
  prsA.io.S := ctl.io.S
  for (b <- 0 until P) prsA.io.din(b) := bspm.io.rData(b)

  val stageBase = (k << (io.nCur - 1.U)).asUInt
  for (t <- 0 until cfg.pBu) {
    val cw = cram.read((ctl.io.coeff(t) + stageBase + io.coeffBase)(log2Ceil(cfg.coeffDepth) - 1, 0))
    bus(t).io.a := prsA.io.dout(2 * t)
    bus(t).io.b := prsA.io.dout(2 * t + 1)
    bus(t).io.c00 := cw(0); bus(t).io.c01 := cw(1)
    bus(t).io.c10 := cw(2); bus(t).io.c11 := cw(3)
    bus(t).io.scale := io.scale
  }

  prsB.io.R := ctl.io.R
  prsB.io.S := ctl.io.S
  for (t <- 0 until cfg.pBu) {
    prsB.io.xin(2 * t)     := bus(t).io.out0
    prsB.io.xin(2 * t + 1) := bus(t).io.out1
  }

  // ---- write port: commit in sWrite (engine), or debug write when idle ----
  val dbgWbank  = FlexBits.bsm(io.dbgWidx, m)
  val dbgWdepth = (io.dbgWidx >> m.U).asUInt
  for (b <- 0 until P) {
    val engWrite = state === sWrite
    val debugW   = io.dbgWen && (state === sIdle) && (b.U === dbgWbank)
    bspm.io.wEn(b)   := engWrite || debugW
    bspm.io.wAddr(b) := Mux(engWrite, ctl.io.addr(b), dbgWdepth)
    bspm.io.wData(b) := Mux(engWrite, prsB.io.dout(b), io.dbgWval)
  }

  io.dbgR := ctl.io.R
  io.dbgS := ctl.io.S
}
