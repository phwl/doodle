// Diagnostic: drive a single ButterflyUnit with known operands and print the
// hardware result, to locate the sign error empirically instead of by reasoning.
package flexbe

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ButterflyDiagSpec extends AnyFlatSpec with Matchers {
  "ButterflyUnit" should "report outputs for known operands" in {
    try {
      simulate(new ButterflyUnit(1, 15)) { c =>
        def q(d: Double) = math.round(d * 32768).toInt
        // out0 = c00*a + c01*b ; out1 = c10*a + c11*b
        // Test 1: identity add, no scale:  a=0.25, b=0.5, c00=c01=1, c10=1,c11=-1
        c.io.a.re.poke(q(0.25).S); c.io.a.im.poke(0.S)
        c.io.b.re.poke(q(0.5).S);  c.io.b.im.poke(0.S)
        c.io.c00.re.poke(q(0.999969).S); c.io.c00.im.poke(0.S)
        c.io.c01.re.poke(q(0.999969).S); c.io.c01.im.poke(0.S)
        c.io.c10.re.poke(q(0.999969).S); c.io.c10.im.poke(0.S)
        c.io.c11.re.poke((-q(0.999969)).S); c.io.c11.im.poke(0.S)
        c.io.scale.poke(false.B)
        c.clock.step()
        println(s"[NOSCALE] out0.re=${c.io.out0.re.peek().litValue}  " +
                s"out1.re=${c.io.out1.re.peek().litValue}  (expect out0~24575, out1~-8192)")

        c.io.scale.poke(true.B)
        c.clock.step()
        println(s"[SCALE ] out0.re=${c.io.out0.re.peek().litValue}  " +
                s"out1.re=${c.io.out1.re.peek().litValue}  (expect out0~12287, out1~-4096)")

        // Test 2: negative operand through c11 (the suspect path)
        c.io.a.re.poke(q(0.5).S);  c.io.a.im.poke(0.S)
        c.io.b.re.poke(q(0.25).S); c.io.b.im.poke(0.S)
        c.io.scale.poke(false.B)
        c.clock.step()
        println(s"[NEG   ] out1.re=${c.io.out1.re.peek().litValue}  " +
                s"(c10*0.5 + c11*0.25 = 0.5 - 0.25 = 0.25 -> ~8192)")
      }
    } catch {
      case e: Exception if e.getMessage != null && e.getMessage.contains("Unexpected message") =>
        cancel("backend unavailable")
    }
  }
}
