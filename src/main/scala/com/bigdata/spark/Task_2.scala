//package com.bigdata.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, MinHashLSH, RegexTokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


object Task_2 {
  def main(args: Array[String]): Unit = {

    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val ss = SparkSession.builder().master("local[2]").appName("task1").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    val inputFile = "./sample_preprocessed.csv"
    val rawDF = ss.read.option("header", "true").csv(inputFile)
    val tempDF = rawDF.select("member_name", "speech_processed").na.drop(Seq("member_name"))

    val df = tempDF.groupBy("member_name").agg(concat_ws(" ", collect_list("speech_processed")) as "speeches")

    val tokenizer = new RegexTokenizer()
      .setPattern(" ")
      .setInputCol("speeches")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(df)

    val vectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(5000)
      //.setMinDF(3)
      .fit(tokenized_df)

    val vectorizedDF = vectorizer.transform(tokenized_df).drop("speeches", "tokens")

    val idf = new IDF()
      .setInputCol("features")
      .setOutputCol("features_idf")
      .fit(vectorizedDF)

    val inverseDF = idf.transform(vectorizedDF)

    def addColumnIndex(df: DataFrame) = {
      ss.sqlContext.createDataFrame(
        df.rdd.zipWithIndex.map {
          case (row, index) => Row.fromSeq(row.toSeq :+ index)
        },
        // Create schema for index column
        StructType(df.schema.fields :+ StructField("index", LongType, nullable = false)))
    }

    val inverseDFIndex = addColumnIndex(inverseDF).drop("member_name", "features")

    inverseDFIndex.show(false)

    val mh = new MinHashLSH()
      .setNumHashTables(500000)
      .setInputCol("features_idf")
      .setOutputCol("hashes")

    val model = mh.fit(inverseDFIndex)

    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(inverseDFIndex).show()

    println("Approximately Jaccard distance smaller than 0.2:")
    model.approxSimilarityJoin(inverseDFIndex, inverseDFIndex, .2, "JaccardDistance")
      .select(col("datasetA.index").alias("idA"),
        col("datasetB.index").alias("idB"),
        col("JaccardDistance")).where(col("idA") =!= col("idB"))
      .sort(col("JaccardDistance"))
      .show()

  }
}
