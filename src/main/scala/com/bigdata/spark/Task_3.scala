import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Task_3 {
  def main(args: Array[String]): Unit = {
    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val sc: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("task3")
      .getOrCreate() // Create Spark Session

    val initial_data = sc // Read the data and create a column Year that contains only the year from the date
      .read
      .option("header", value = true)
      .csv("data_dropna_rm_morestop.csv")
      .withColumn("Year", split(col("sitting_date"), "/")
        .getItem(2))
      .drop(col("sitting_date"))

    val years = initial_data //take all distinct years as a list
      .select("Year")
      .distinct()
      .sort(col("Year").asc)
      .rdd
      .collect()
      .toList

    val party = initial_data //take all distinct political party's as a list
      .select("political_party")
      .distinct()
      .rdd
      .collect()
      .toList

    val members = initial_data //take all distinct political party's as a list
      .select("member_name")
      .distinct()
      .rdd
      .collect()
      .toList

    for (m <- members) {
      for (y <- years) { // for each single year find its keywords
        println("Keywords for member", m(0), "and year", y(0), "are:")
        find_keywords("member_name",m(0), "Year", y(0))
      }
    }

    for (p <- party) { // for each single party find its keywords
      for (y <- years) {
        println("Keywords for party", p(0), "and year", y(0), "are:")
        find_keywords("political_party", p(0), "Year", y(0))
      }
    }

    def find_keywords(c: String, value: Any, c1: String, value1: Any): Unit = { // function that finds keywords

      val preprocessed_speeches = initial_data // choose the data
        .where(col(c) === value && col(c1) === value1)
        .select(split(col("speech_processed"), " "))
        .withColumn("doc_id", monotonically_increasing_id())
        .withColumnRenamed("split(speech_processed,  , -1)", "speech_processed")

      val columns = preprocessed_speeches // take all words from preprocessed spechees as tokens
        .columns
        .map(col) :+
        (explode(col("speech_processed")) as "token")
      val unfoldedDocs = preprocessed_speeches
        .select(columns: _*)

      val docCount = unfoldedDocs // Compute in how many documents each word is shown
        .select("doc_id")
        .distinct()
        .count()

      // Dataframes with term frequency and document frequency
      val tokensWithTf = unfoldedDocs
        .groupBy("doc_id", "token")
        .agg(count("speech_processed") as "tf")
      val unfoldedDocs_2 = unfoldedDocs
        .groupBy("token")
        .agg(countDistinct("doc_id") as "df")

      def calcIdf(docCount: Long, df: Long): Double = { //Calculate IDF
        val Idf_type = math
          .log(((docCount + 1) / (df + 1))
            .toDouble)
        Idf_type
      }

      val calcIdfUdf = udf { df: Long => calcIdf(docCount, df) }

      val tokensWithIdf = unfoldedDocs_2 //Dataframe with idfs
        .withColumn("idf", calcIdfUdf(col("df")))

      val tokensWith_tf_Idf = tokensWithTf //Dataframe with tf*idf
        .join(tokensWithIdf, Seq("token"), "left")
        .withColumn("tf_idf", col("tf") * col("idf"))

      val results = tokensWith_tf_Idf // Show results
        .withColumnRenamed("token", "Keywords")
        .groupBy("Keywords")
        .sum("tf_idf")
        .orderBy(col("sum(tf_idf)")
          .desc)
        .select("Keywords")

      results.show(10, truncate = false)
    }
  }
}
