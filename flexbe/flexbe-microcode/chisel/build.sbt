// FlexBE microcoded command processor -- Chisel build.
//
//   sbt test        run the ChiselSim tests (checks bsm / PRS / cycle_control
//                   against the same golden vectors as the Python model, and
//                   runs a 16-point FFT through the datapath)
//   sbt "runMain flexbe.Emit"   emit SystemVerilog to generated/
//
// Requires JDK 17+ and network access to Maven Central on first build.

ThisBuild / scalaVersion := "2.13.14"
ThisBuild / organization := "flexbe"
ThisBuild / version      := "0.1.0"

lazy val root = (project in file("."))
  .settings(
    name := "flexbe-chisel",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel"      % "6.6.0",
      "org.scalatest"     %% "scalatest"   % "3.2.19" % "test"
    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % "6.6.0" cross CrossVersion.full),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations"
    )
  )
