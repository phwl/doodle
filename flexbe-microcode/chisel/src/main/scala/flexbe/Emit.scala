// Emit SystemVerilog for the main blocks.  `sbt "runMain flexbe.Emit"`.
package flexbe

import circt.stage.ChiselStage

object Emit extends App {
  val dir = "generated"
  def emit(name: String, gen: () => chisel3.RawModule): Unit = {
    ChiselStage.emitSystemVerilogFile(
      gen(),
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info",
                          "-o", s"$dir/$name.sv"))
    println(s"wrote $dir/$name.sv")
  }
  emit("Engine",    () => new Engine(new EngineConfig()))
  emit("Sequencer", () => new Sequencer(new SequencerConfig()))
  emit("PRS",       () => new PRS(32))
  emit("CycleControl", () => new CycleControl(15, 16))
}
