package com.bigdata.spark

import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{collect_list, concat_ws}

object task2 {
  val ss = SparkSession.builder().master("local[2]").appName("task1").getOrCreate()
  ss.sparkContext.setLogLevel("ERROR")

  import ss.implicits._

  val inputFile = "./sample_preprocessed.csv"
  val rawDF = ss.read.option("header", "true").csv(inputFile)
  val tempDF = rawDF.select("member_name", "speech_processed").na.drop(Seq("member_name"))

  val df = tempDF.groupBy("member_name").agg(concat_ws(" ", collect_list("speech_processed")) as "speeches")

  df.printSchema()
  df.show(20)

  val tokenizer = new RegexTokenizer()
    .setPattern(" ")
    .setInputCol("speeches")
    .setOutputCol("tokens")

  val tokenized_df = tokenizer.transform(df)

  tokenized_df.printSchema()
  tokenized_df.show(10)

  val vectorizer = new CountVectorizer()
    .setInputCol("tokens")
    .setOutputCol("features")
    .setVocabSize(5000)
    //.setMinDF(3)
    .fit(tokenized_df)

  val vectorizedDF = vectorizer.transform(tokenized_df).drop("speeches", "tokens")

  vectorizedDF.printSchema()
  vectorizedDF.show(25)

}
