ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.11.12"

lazy val root = (project in file("."))
  .settings(
    name := "NeRF"
  )

libraryDependencies += "ai.djl" % "api" % "latest.release"
libraryDependencies += "ai.djl.mxnet" % "mxnet-engine" % "latest.release" % "runtime"
libraryDependencies += "ai.djl.mxnet" % "mxnet-native-auto" % "latest.release" % "runtime"
libraryDependencies += "ai.djl" % "model-zoo" % "latest.release"
libraryDependencies += "ai.djl" % "basicdataset" % "latest.release"

scalacOptions ++= Seq(
  "-target:jvm-1.8",
  "-Dhttp.proxyHost=127.0.0.1",
  "-Dhttp.proxyPort=10809"
)

libraryDependencies += "com.github.scopt" % "scopt_2.11" % "latest.release"
//libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "latest.release"
