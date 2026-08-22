// =============================================================================
// BankedMemory.scala -- the P-bank butterfly scratchpad (BSPM).
//
// P = 2*P_bu single-port banks, each `depth` words.  Every cycle the engine
// reads one word from every bank and writes one word to every bank, each bank
// at its own address (the two-row regime of CycleControl reads two depths at
// once, so a single shared address will not do).  The shift-down skew of
// Eq. (3) is applied by whoever supplies the addresses -- software addresses
// BSPM by logical element index and never sees the banking.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

class BankPort(val depthW: Int) extends Bundle {
  val addr = UInt(depthW.W)
}

class BankedMemory(val P: Int, val depth: Int,
                   val intBits: Int = 1, val fracBits: Int = 15) extends Module {
  val depthW = log2Ceil(depth)
  val io = IO(new Bundle {
    val rAddr = Input(Vec(P, UInt(depthW.W)))
    val rData = Output(Vec(P, new Cplx(intBits, fracBits)))
    val wEn   = Input(Bool())
    val wAddr = Input(Vec(P, UInt(depthW.W)))
    val wData = Input(Vec(P, new Cplx(intBits, fracBits)))
  })

  val banks = Seq.fill(P)(SyncReadMem(depth, new Cplx(intBits, fracBits)))
  for (b <- 0 until P) {
    io.rData(b) := banks(b).read(io.rAddr(b))
    when(io.wEn) { banks(b).write(io.wAddr(b), io.wData(b)) }
  }
}
