// =============================================================================
// Sequencer.scala -- the microcoded command processor of Sec. 4.
//
// A three-stage front end for the fixed-function engine: fetch descriptors from
// a ring in coherent memory, decode each into engine control, and dispatch it
// into the queue of its resource class (compute, DMA-in, DMA-out).  Commands
// issue in program order per resource and start only once the tags they depend
// on have retired; a tag scoreboard of done-bits makes the dependency check a
// lookup rather than an associative search.  Completion is by memory: the
// retiring command writes its tag to a completion port the host polls.
//
// This is control-plane RTL.  The actual datapath timing (Eq. 10 cycle counts)
// is produced by the Engine; here each dispatched command carries a cycle
// budget that the resource model counts down, so the module reproduces the
// overlap behaviour the Python Sequencer models.
// =============================================================================
package flexbe

import chisel3._
import chisel3.util._

/** A command as issued to a resource: its cycle cost and its dependency tag. */
class IssuedCmd(val tagW: Int) extends Bundle {
  val desc     = new Descriptor
  val cycles   = UInt(32.W)          // engine cycles (compute) or transfer beats
  val depTag   = UInt(tagW.W)        // last tag that must retire first
  val hasDep   = Bool()
}

class SequencerConfig(
  val ringSlots: Int = 16,
  val tagW: Int = 8,
  val prefetch: Int = 8
)

/** One resource lane: a small queue plus a countdown timer for the head. */
class ResourceLane(cfg: SequencerConfig) extends Module {
  val io = IO(new Bundle {
    val enq     = Flipped(Decoupled(new IssuedCmd(cfg.tagW)))
    val doneVec = Input(Vec(1 << cfg.tagW, Bool()))  // scoreboard snapshot
    val retire  = Valid(UInt(cfg.tagW.W))            // tag that just completed
    val busy    = Output(Bool())
  })

  val q = Module(new Queue(new IssuedCmd(cfg.tagW), cfg.ringSlots))
  q.io.enq <> io.enq

  val counter = RegInit(0.U(32.W))
  val running = RegInit(false.B)
  val curTag  = Reg(UInt(cfg.tagW.W))

  io.retire.valid := false.B
  io.retire.bits  := curTag
  io.busy         := running || q.io.deq.valid

  q.io.deq.ready := false.B
  when(running) {
    when(counter === 1.U) {
      running := false.B
      io.retire.valid := true.B          // retire on the last cycle
    }.otherwise {
      counter := counter - 1.U
    }
  }.elsewhen(q.io.deq.valid) {
    val head = q.io.deq.bits
    val depOk = !head.hasDep || io.doneVec(head.depTag)
    when(depOk) {
      q.io.deq.ready := true.B
      curTag  := head.desc.tag(cfg.tagW - 1, 0)
      counter := head.cycles
      running := head.cycles =/= 0.U
      when(head.cycles === 0.U) { io.retire.valid := true.B }
    }
  }
}

/** The command processor: decode + dispatch + scoreboard + completion. */
class Sequencer(cfg: SequencerConfig = new SequencerConfig) extends Module {
  val io = IO(new Bundle {
    /** raw 256-bit descriptors streamed from the fetch unit, in program order */
    val fetch   = Flipped(Decoupled(UInt(256.W)))
    /** cycle budget accompanying each descriptor (from the engine model) */
    val budget  = Input(UInt(32.W))
    /** completion: tag of a command that has retired */
    val complete = Valid(UInt(cfg.tagW.W))
    /** dispatch handles to the engine / DMA (decoded head of each lane) */
    val idle    = Output(Bool())
  })

  // ---- scoreboard: done-bit per tag ---------------------------------------
  val done = RegInit(VecInit(Seq.fill(1 << cfg.tagW)(false.B)))

  // ---- three resource lanes -----------------------------------------------
  val lanes = Seq.fill(3)(Module(new ResourceLane(cfg)))
  for (l <- lanes) l.io.doneVec := done

  // ---- decode + dispatch --------------------------------------------------
  val d = Descriptor.decode(io.fetch.bits)
  val rc = d.resClass

  val issued = Wire(new IssuedCmd(cfg.tagW))
  issued.desc   := d
  issued.cycles := io.budget
  // dependency: a command waits on the immediately preceding tag unless it is
  // the first (tag 0).  The full DAG is expressed by the program's tag order;
  // this models in-order-per-resource with a single predecessor edge, matching
  // the Python model's `deps` handling for the linear BSPNet chain.
  issued.depTag := Mux(d.tag === 0.U, 0.U, (d.tag - 1.U)(cfg.tagW - 1, 0))
  issued.hasDep := d.tag =/= 0.U

  // route fetch to the selected lane
  io.fetch.ready := false.B
  for ((l, i) <- lanes.zipWithIndex) {
    l.io.enq.valid := io.fetch.valid && (rc === i.U)
    l.io.enq.bits  := issued
    when(rc === i.U) { io.fetch.ready := l.io.enq.ready }
  }

  // ---- retirement: update scoreboard, emit completion ---------------------
  val anyRetire = lanes.map(_.io.retire.valid).reduce(_ || _)
  val retTag    = MuxCase(0.U, lanes.map(l => l.io.retire.valid -> l.io.retire.bits))
  when(anyRetire) { done(retTag) := true.B }
  io.complete.valid := anyRetire
  io.complete.bits  := retTag

  io.idle := !io.fetch.valid && lanes.map(!_.io.busy).reduce(_ && _)
}
