// =============================================================================
// Descriptor.scala -- the 32-byte microcode command word.
//
// Byte layout, little-endian, matching Descriptor.encode() in microcode.py
// (struct '<8B6I'):
//   byte 0  opcode      byte 1  flags       byte 2  mode      byte 3  log2_l
//   byte 4  log2_Psub   byte 5  stage_lo    byte 6  stage_hi  byte 7  pool
//   word 2  src         word 3  dst         word 4  coeff
//   word 5  count       word 6  P_N         word 7  tag
//
// Opcodes and flags are the same constants as the Python assembler, so a
// descriptor stream produced there loads directly into this hardware.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

object Op {
  val LOAD  = 0x01
  val WLOAD = 0x02
  val STORE = 0x03
  val BFLY  = 0x10
  val BREV  = 0x11
  val POW   = 0x20
  val MAG   = 0x21
  val POOL  = 0x22
  val all = Seq(LOAD, WLOAD, STORE, BFLY, BREV, POW, MAG, POOL)
}

object Mode { val FFT = 0; val IFFT = 1; val BL = 2; val BL_EXPAND = 3 }

object Flag {
  val NATURAL   = 0x00
  val BITREV    = 0x01
  val MAG       = 0x02
  val NORMALISE = 0x04
  val RELU      = 0x08
  val LAYERNORM = 0x10
  val SHORTCUT  = 0x20
  val MAXPOOL   = 0x40
}

object ResClass {
  val COMPUTE = 0
  val DMA_IN  = 1
  val DMA_OUT = 2
  val W = 2
}

/** Decoded descriptor fields. */
class Descriptor extends Bundle {
  val opcode   = UInt(8.W)
  val flags    = UInt(8.W)
  val mode     = UInt(8.W)
  val log2L    = UInt(8.W)
  val log2Psub = UInt(8.W)
  val stageLo  = UInt(8.W)
  val stageHi  = UInt(8.W)
  val pool     = UInt(8.W)
  val src      = UInt(32.W)
  val dst      = UInt(32.W)
  val coeff    = UInt(32.W)
  val count    = UInt(32.W)
  val pN       = UInt(32.W)
  val tag      = UInt(32.W)

  /** which resource queue this command belongs to */
  def resClass: UInt = {
    val c = Wire(UInt(ResClass.W.W))
    c := ResClass.COMPUTE.U
    switch(opcode) {
      is(Op.LOAD.U)  { c := ResClass.DMA_IN.U }
      is(Op.WLOAD.U) { c := ResClass.DMA_IN.U }
      is(Op.STORE.U) { c := ResClass.DMA_OUT.U }
    }
    c
  }
  def isCompute: Bool = resClass === ResClass.COMPUTE.U
  def flagSet(bit: Int): Bool = (flags & bit.U) =/= 0.U
}

object Descriptor {
  /** Decode a 256-bit little-endian word (word0 in bits [31:0]) into fields. */
  def decode(raw: UInt): Descriptor = {
    require(raw.getWidth == 256)
    val d = Wire(new Descriptor)
    d.opcode   := raw(7, 0)
    d.flags    := raw(15, 8)
    d.mode     := raw(23, 16)
    d.log2L    := raw(31, 24)
    d.log2Psub := raw(39, 32)
    d.stageLo  := raw(47, 40)
    d.stageHi  := raw(55, 48)
    d.pool     := raw(63, 56)
    d.src      := raw(95, 64)
    d.dst      := raw(127, 96)
    d.coeff    := raw(159, 128)
    d.count    := raw(191, 160)
    d.pN       := raw(223, 192)
    d.tag      := raw(255, 224)
    d
  }
}
