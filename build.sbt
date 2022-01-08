name := "bigdataSpark"

version := "0.1"

scalaVersion := "2.12.8"
//scalaVersion := "3.0.1"


//val sparkVersion = "3.0.3"
val sparkVersion = "3.2.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion

  //"org.apache.spark" %% "spark-streaming" % sparkVersion,
  //"org.apache.spark" %% "spark-hive" % sparkVersion
)

