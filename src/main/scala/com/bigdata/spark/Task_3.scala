//package com.bigdata.spark
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import scala.language.postfixOps


object Task_3 {
  def main(args: Array[String]): Unit = {

    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val sc: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate()

    val initial_data = sc
      .read
      .option("header", value = true)
      .csv("sample_preprocessed.csv")
      .withColumn("Year", split(col("sitting_date"), "/")
      .getItem(2))
      .drop(col("sitting_date"))

    val years = initial_data
      .select("Year")
      .distinct()
      .rdd
      .collect()
      .toList

    val party = initial_data
      .select("political_party")
      .distinct()
      .rdd
      .collect()
      .toList


    for (element <- years){
      find_keywords("Year", element(0))
    }

    for (element <- party){
      find_keywords("political_party", element(0))
    }

    def find_keywords(c:String, value: Any): Unit = {

      val preprocessed_speeches = initial_data
        .select(split(col("speech_processed"), " "))
        .where(col(c) === value)
        .withColumn("doc_id", monotonically_increasing_id())
        .withColumnRenamed("split(speech_processed,  , -1)", "speech_processed")

      val columns = preprocessed_speeches
        .columns
        .map(col) :+
        (explode(col("speech_processed")) as "token")
      val unfoldedDocs = preprocessed_speeches
        .select(columns: _*)

      val docCount = unfoldedDocs
        .select("doc_id")
        .distinct()
        .count()

      val tokensWithTf = unfoldedDocs
        .groupBy("doc_id", "token")
        .agg(count("speech_processed") as "tf")
      val unfoldedDocs_2 = unfoldedDocs
        .groupBy("token")
        .agg(countDistinct("doc_id") as "df")


      def calcIdf(docCount: Long, df: Long): Double = {
        val Idf_type = math
          .log(((docCount + 1) / (df + 1))
            .toDouble)
        Idf_type
      }

      val calcIdfUdf = udf { df: Long => calcIdf(docCount, df) }

      val tokensWithIdf = unfoldedDocs_2
        .withColumn("idf", calcIdfUdf(col("df")))

      val tokensWith_tf_Idf = tokensWithTf
        .join(tokensWithIdf, Seq("token"), "left")
        .withColumn("tf_idf", col("tf") * col("idf"))

      tokensWith_tf_Idf
        .withColumnRenamed("token", "Keywords")
        .orderBy(col("tf_idf")
          .desc)
        .select("Keywords")
        .show(15)
    }
  }
}
