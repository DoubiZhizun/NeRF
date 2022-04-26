ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.11.12"

lazy val root = (project in file("."))
  .settings(
    name := "NeRF"
  )

libraryDependencies += "ai.djl" % "api" % "0.16.0"
libraryDependencies += "ai.djl.pytorch" % "pytorch-engine" % "0.16.0" % "runtime"
libraryDependencies += "ai.djl.pytorch" % "pytorch-native-cpu" % "1.8.1" % "runtime"
libraryDependencies += "ai.djl.pytorch" % "pytorch-jni" % "1.8.1-0.16.0" % "runtime"
libraryDependencies += "ai.djl" % "model-zoo" % "latest.release"
libraryDependencies += "ai.djl" % "basicdataset" % "latest.release"

scalacOptions ++= Seq(
  "-target:jvm-1.8",
  "-Dhttp.proxyHost=127.0.0.1",
  "-Dhttp.proxyPort=10809"
)

libraryDependencies += "com.github.scopt" % "scopt_2.11" % "latest.release"
//libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "latest.release"
