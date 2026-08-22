// =============================================================================
// BankedMemory.scala -- the P-bank butterfly scratchpad (BSPM).
//
// P = 2*P_bu genuinely single-port banks, each `depth` words.  Every cycle the
// engine reads one word from every bank and writes one word to every bank, each
// bank at its own address and with its own write strobe (the two-row regime of
// CycleControl reads two depths at once, so a single shared address will not
// do; per-bank write-enables also let a testbench preload one location at a
// time through these same ports).  The shift-down skew of Eq. (3) is applied by
// whoever supplies the addresses -- software addresses BSPM by logical element
// index and never sees the banking.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

class BankedMemory(val P: Int, val depth: Int,
                   val intBits: Int = 1, val fracBits: Int = 15) extends Module {
  val depthW = log2Ceil(depth)
  val io = IO(new Bundle {
    val rAddr = Input(Vec(P, UInt(depthW.W)))
    val rData = Output(Vec(P, new Cplx(intBits, fracBits)))
    val wEn   = Input(Vec(P, Bool()))            // per-bank write strobe
    val wAddr = Input(Vec(P, UInt(depthW.W)))
    val wData = Input(Vec(P, new Cplx(intBits, fracBits)))
  })

  val banks = Seq.fill(P)(SyncReadMem(depth, new Cplx(intBits, fracBits)))
  for (b <- 0 until P) {
    io.rData(b) := banks(b).read(io.rAddr(b))
    when(io.wEn(b)) { banks(b).write(io.wAddr(b), io.wData(b)) }
  }
}
