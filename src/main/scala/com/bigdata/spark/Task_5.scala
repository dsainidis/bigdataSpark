import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable

object Task_5 {
  def main(args: Array[String]): Unit = {

    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val ss: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("task5")
      .getOrCreate() // Create Spark Session

    // Read input file, select the speeches and the date and create a column that contain only the years as we dont need
    // the rest date
    val inputFile = "./data_dropna_rm_morestop.csv"
    val inputDF = ss
      .read
      .option("header", "true")
      .csv(inputFile)
    val tempDF = inputDF
      .select("member_name", "speech_processed")

    val df = tempDF //
      .withColumn("id", monotonically_increasing_id())
      .drop("sitting_date")
      .na
      .drop(Seq("speech_processed"))

    val tokenizer = new RegexTokenizer() // Tokenizer
      .setPattern(" ")
      .setInputCol("speech_processed")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer
      .transform(df)

    val vectorizer = new CountVectorizer() // CountVectorizer
      .setInputCol("tokens")
      .setOutputCol("rawFeatures")
      .setVocabSize(5000)
      .setMinDF(3)
      .fit(tokenized_df)

    val vectorizedDF = vectorizer.transform(tokenized_df)
    val vocab = vectorizer.vocabulary //vocab should be broadcasted

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(vectorizedDF)

    val inverseDF = idf.transform(vectorizedDF)

    def find_topics(df: DataFrame): DataFrame = { // function to find topics. It is same as task 1
      val corpus = df.select("id", "features")

      val lda = new LDA()
        .setSeed(0)
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

      val termIndicesToWords = udf((x: mutable.WrappedArray[Int]) => {
        x.map(i => vocab(i))
      })

      println("The topics described by their top-weighted terms:")
      val topics = rawTopics.withColumn("topicWords", termIndicesToWords(col("termIndices")))
      topics
    }

    // find the topics for all years and take each word separately as a dataframe column
    println("All year topics")
    val topics = find_topics(inverseDF)
    val topic_words = topics.select("topic", "termWeights", "topicWords")
    val topic_words_explode = topic_words.columns
      .map(col) :+
      (explode(col("topicWords")) as "token")
    val topic_words_explode_ = topics.select(topic_words_explode: _*).drop("topicWords")
    topic_words.show(false)

    val initial_data = ss // read the data again
      .read
      .option("header", value = true)
      .csv("./data_dropna_rm_morestop.csv")

    val members = initial_data // take all member names
      .select(col("member_name"))
      .distinct()
      .rdd
      .collect()
      .toList

    def count_words(value: Any, colum: String): Unit = { // function that simply counts the words from data

      val preprocessed_speeches = initial_data // select data
        .where(col(colum) === value)
        .select(split(col("speech_processed"), " "))
        .withColumnRenamed("split(speech_processed,  , -1)", "speech_processed")

      val columns = preprocessed_speeches // take all words from preprocessed spechees as tokens
        .columns
        .map(col) :+
        (explode(col("speech_processed")) as "token")
      val unfoldedDocs = preprocessed_speeches
        .select(columns: _*)

      val tokensWithTf = unfoldedDocs // Dataframe that contains the word frequency descending sorted
        .groupBy("token")
        .agg(count("token") as "tf")
        .sort(col("tf").desc)

      val results = tokensWithTf // create the results by joining the topic words and the member's or party's words
        // and count them with respect to the topic each word belongs
        .join(topic_words_explode_, tokensWithTf("token") === topic_words_explode_("token"), "inner")
        .groupBy("topic")
        .sum("tf")
        .sort(col("sum(tf)").desc)
        .withColumnRenamed("sum(tf)", "words")

      results.show()
    }

    members.foreach(i=>{ // for all members compute most frequent words
      println("Most used words of member", i, "are")
      count_words(i(0), "member_name")
    })

    val partys = initial_data // take all party's to a list
      .select(col("political_party"))
      .distinct()
      .rdd
      .collect()
      .toList

    partys.foreach(i=>{ // for all party's compute most frequent words
      println("Most used words of party", i, "are")
      count_words(i(0), "political_party")
    })
  }
}
