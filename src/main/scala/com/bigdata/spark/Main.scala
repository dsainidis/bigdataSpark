package com.bigdata.spark

import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    // Create the spark session first
    val ss = SparkSession.builder().master("local[2]").appName("task1").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    import ss.implicits._

    val inputFile = "./sample_preprocessed.csv"

    println("reading from input file: " + inputFile)
    println

    // Read the contents of the csv file in a dataframe.
    val rawDF = ss.read.option("header", "true").csv(inputFile)
    rawDF.printSchema()

    val cols = Seq("_c0","member_name","parliamentary_period", "parliamentary_session", "parliamentary_sitting",
      "political_party", "government", "member_region", "roles", "member_gender", "speech")

    val df = rawDF.drop(cols:_*)
    df.printSchema()

    val sitting_dates = df.select("sitting_date").rdd.map(r => r(0).asInstanceOf[String]).collect()






  }
}
