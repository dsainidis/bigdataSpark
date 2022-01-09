package com.bigdata.spark

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, udf}

import scala.collection.mutable

object Task_1 {

  val ss = SparkSession.builder().master("local[2]").appName("task1").getOrCreate()
  ss.sparkContext.setLogLevel("ERROR")

  import ss.implicits._

  val inputFile = "./sample_preprocessed.csv"
  val inputDF = ss.read.option("header", "true").csv(inputFile)
  val tempDF = inputDF.select("sitting_date", "speech_processed")
  val yearColumn = tempDF.select("sitting_date").rdd.map(x => x.toString().split("/")(2).substring(0, 4).toInt).toDF("year")

  val df = tempDF.withColumn("id", monotonically_increasing_id()).join(yearColumn).drop("sitting_date").na.drop(Seq("speech_processed"))

  df.printSchema()
  df.show(10)

  val tokenizer = new RegexTokenizer()
    .setPattern(" ")
    .setInputCol("speech_processed")
    .setOutputCol("tokens")

  val tokenized_df = tokenizer.transform(df)

  tokenized_df.printSchema()
  tokenized_df.show(10)

  val vectorizer = new CountVectorizer()
    .setInputCol("tokens")
    .setOutputCol("rawFeatures")
    .setVocabSize(10000)
    .setMinDF(3)
    .fit(tokenized_df)

  val vectorizedDF = vectorizer.transform(tokenized_df)
  val vocab = vectorizer.vocabulary  //vocab should be broadcasted

  vectorizedDF.printSchema()
  vectorizedDF.show(10)

  val idf = new IDF()
    .setInputCol("rawFeatures")
    .setOutputCol("features")
    .fit(vectorizedDF)

  val inverseDF = idf.transform(vectorizedDF)

  inverseDF.printSchema()
  inverseDF.show(10)

  val corpus = inverseDF.select("id", "features")
  corpus.show(10)

  val lda = new LDA()
    .setOptimizer("em")
    .setK(10)
    .setMaxIter(50)

  val ldaModel = lda.fit(corpus)

  val ll = ldaModel.logLikelihood(corpus)
  val lp = ldaModel.logPerplexity(corpus)
  println(s"The lower bound on the log likelihood of the entire corpus: $ll")
  println(s"The upper bound on perplexity: $lp")

  // Describe topics.
  val rawTopics = ldaModel.describeTopics(10)
  rawTopics.printSchema()

  val termIndicesToWords = udf( (x : mutable.WrappedArray[Int]) => { x.map(i => vocab(i)) })

  println("The topics described by their top-weighted terms:")
  val topics = rawTopics.withColumn("topicWords", termIndicesToWords(col("termIndices")))
  topics.printSchema()
  topics.select("topic", "topicWords").show(10, truncate = false)

  // Shows the result.
  val transformed = ldaModel.transform(corpus)
  transformed.show(false)
}
