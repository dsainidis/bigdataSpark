package com.bigdata.spark

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.{EMLDAOptimizer, LDA, OnlineLDAOptimizer}

import scala.language.postfixOps

object Main {
  def main(args: Array[String]): Unit = {
    val ss = SparkSession.builder().master("local[2]").appName("task1").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    import ss.implicits._

    val inputFile = "./sample_preprocessed.csv"
    val rawDF = ss.read.option("header", "true").csv(inputFile)
    val tempDF = rawDF.select("sitting_date", "speech_processed")
    val yearDF = tempDF.select("sitting_date").rdd.map(x => x.toString().split("/")(2).substring(0, 4).toInt).toDF("year")

    val df = tempDF.withColumn("id", monotonically_increasing_id()).join(yearDF).drop("sitting_date").na.drop(Seq("speech_processed"))

    df.printSchema()
    df.show(10)

    val tokenizer = new RegexTokenizer()
      .setPattern(" ")
      .setInputCol("speech_processed")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(df).drop("speech_processed", "year")

    tokenized_df.printSchema()
    tokenized_df.show(10)

    val vectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(10000)
      //.setMinDF(5)
      .fit(tokenized_df)

    val countVectors = vectorizer.transform(tokenized_df).select("id", "features")

    countVectors.printSchema()
    countVectors.show(20)

    val lda_countVector = countVectors.map { case Row(id: Long, countVector: Vector) => (id, countVector) }

    lda_countVector.printSchema()

    val lda = new LDA()

    lda.setOptimizer(new EMLDAOptimizer())
      .setK(50)
      .setMaxIterations(3)
      .setDocConcentration(-1) // use default values
      .setTopicConcentration(-1) // use default values

  }
}

