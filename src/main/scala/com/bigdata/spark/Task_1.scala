package org.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, split, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable

object Task_1 {
  def main(args: Array[String]): Unit = {
    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val ss: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("task1")
      .getOrCreate() // Create Spark Session

    // Read input file, select the speeches and the date and create a column that contain only the years as we dont need
    // the rest date
    val inputFile = "./sample_preprocessed.csv"
    val inputDF = ss
      .read
      .option("header", "true")
      .csv(inputFile)
    val tempDF = inputDF
      .select("sitting_date", "speech_processed")
      .withColumn("Year", split(col("sitting_date"), "/")
        .getItem(2))

    val df = tempDF //Create a column with increasingly id and drop nulls.
      .withColumn("id", monotonically_increasing_id())
      .drop("sitting_date")
      .na
      .drop(Seq("speech_processed"))

    val tokenizer = new RegexTokenizer() //Create tokenizer that splits base on a pattern
      .setPattern(" ")
      .setInputCol("speech_processed")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer //Tokenize data
      .transform(df)

    val vectorizer = new CountVectorizer() //Use vectorizer to convert text to vector
      .setInputCol("tokens")
      .setOutputCol("rawFeatures")
      .setVocabSize(20000)
      .setMinDF(3)
      .fit(tokenized_df)

    val vectorizedDF = vectorizer
      .transform(tokenized_df) //Vectorize data
    val vocab = vectorizer
      .vocabulary //vocab should be broadcasted

    val idf = new IDF() // Compute IDF of rawFeatures from CountVectorizer
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(vectorizedDF)

    val inverseDF = idf
      .transform(vectorizedDF) // Transform

    val years = inverseDF //take all distinct years as a list
      .select("Year")
      .distinct()
      .rdd
      .collect()
      .toList

    println("All years topics")
    find_topics(inverseDF)

    for (element <- years){ // for each single year find topics and its keywords
      val tempDF = inverseDF.where(col("Year") === element(0))
      println("Year", element(0), "topics are:")
      find_topics(tempDF)
    }

    def find_topics(df:DataFrame): Unit = { // function for finding topics
      val corpus = df
        .select("id", "features")

      val lda = new LDA() //Topics are discovered using LDA clustering
        .setOptimizer("em")
        .setK(10)
        .setMaxIter(50)

      val ldaModel = lda
        .fit(corpus) //Fit the lda model

      val ll = ldaModel
        .logLikelihood(corpus)
      val lp = ldaModel
        .logPerplexity(corpus)
      println(s"The lower bound on the log likelihood of the entire corpus: $ll")
      println(s"The upper bound on perplexity: $lp")

      // Describe topics.
      val rawTopics = ldaModel
        .describeTopics(10)

      val termIndicesToWords = udf((x: mutable
      .WrappedArray[Int]) => { //This function hepls to convert term indices back to words
        x.map(i => vocab(i))
      })

      //save results to a dataframe and print them
      println("The topics described by their top-weighted terms:")
      val topics = rawTopics
        .withColumn("topicWords", termIndicesToWords(col("termIndices")))
      topics
        .select("topic", "termWeights", "topicWords").show(30, truncate = false)
    }
  }
}
