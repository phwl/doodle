ThisBuild / scalaVersion := "2.13.15"

lazy val root = (project in file("."))
  .settings(
    name := "chisel-claude-verilator",
    
    libraryDependencies ++= Seq(
      // Core Chisel framework
      "org.chipsalliance" %% "chisel" % "6.6.0",
      
      // Matched HTTP & JSON libraries matching Chisel's internal tree
      "com.lihaoyi"       %% "requests" % "0.8.0",
      "com.lihaoyi"       %% "upickle"  % "3.1.0",
      
      // Unit testing framework
      "org.scalatest"     %% "scalatest" % "3.2.19" % Test
    ),
    
    // Required macro compiler plugin matching your precise framework version
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % "6.6.0" cross CrossVersion.full)
  )

