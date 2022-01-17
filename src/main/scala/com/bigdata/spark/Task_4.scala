package org.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}


object Task_4 {
  def main(args: Array[String]): Unit = {

    def create_inverse_DF(c:String, data:DataFrame): DataFrame = {

      val grouped_by = data // group data by column that is argument to function
        .groupBy(c)
        .agg(concat_ws(" ", collect_list("speech_processed")) as "speeches")

      val tokenizer = new RegexTokenizer() //Tokenizer
        .setPattern(" ")
        .setInputCol("speeches")
        .setOutputCol("tokens")

      val tokenized_df = tokenizer
        .transform(grouped_by)
        .drop("speech_processed")

      val vectorizer = new CountVectorizer() //CountVectorizer
        .setInputCol("tokens")
        .setOutputCol("features")
        .setVocabSize(20000)
        .setMinDF(3)
        .fit(tokenized_df)

      val vectorizedDF = vectorizer
        .transform(tokenized_df)

      val idf = new IDF()
        .setInputCol("features")
        .setOutputCol("features_idf")
        .fit(vectorizedDF)

      val inverseDF = idf
        .transform(vectorizedDF)

      inverseDF
    }

    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val sc: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("task4")
      .getOrCreate() // Create Spark Session

    val cosSimilarity = udf { (x: Vector, y: Vector) => // Udf that calculates cosine Similarity of vectors
      val v1 = x.toArray
      val v2 = y.toArray
      val l1 = scala.math.sqrt(v1.map(x => x*x).sum)
      val l2 = scala.math.sqrt(v2.map(x => x*x).sum)
      val scalar = v1.zip(v2).map(p => p._1*p._2).sum
      scalar/(l1*l2)
    }

    val initial_data = sc //read the data, create column that conatains only the year and drop nulls
      .read
      .option("header", value = true)
      .csv("sample_preprocessed.csv")
      .withColumn("Year", split(col("sitting_date"), "/")
        .getItem(2))
      .drop(col("sitting_date"))
      .na
      .drop(Seq("member_name"))

    // Separate data for member names before and after the financial crisis and rename some columns
    val before_2009 = initial_data.where(col("Year") < 2009)
    val after_2009 = initial_data.where(col("Year") >= 2009)
    val before_2009_final = create_inverse_DF("member_name", before_2009)
      .withColumnRenamed("features_idf", "features_idf_before_2009")
    val after_2009_final = create_inverse_DF("member_name", after_2009)
      .withColumnRenamed("features_idf", "features_idf_after_2009")


    // Join the smallest between two datasets to avoid nulls and compute cosine similarity between the speeches
    // before and after 2009. Then show the results ascending because small similarity means big distance
    println("Most different pairs of members with respect to speeches are:")
    if (before_2009_final.rdd.count() >= after_2009_final.rdd.count()) {
      val results = after_2009_final
        .join(before_2009_final, Seq("member_name"))
        .withColumn("Similarity", cosSimilarity(col("features_idf_before_2009"), col("features_idf_after_2009")))
        .sort(col("Similarity").asc)
        .select("member_name", "Similarity")
      results.show()
    }else{
      val results = before_2009_final
        .join(after_2009_final, Seq("member_name"))
        .withColumn("Similarity", cosSimilarity(col("features_idf_after_2009"), col("features_idf_before_2009")))
        .sort(col("Similarity").asc)
        .select("member_name", "Similarity")
      results.show()
    }

    // Separate data for political party's before and after the financial crisis and rename some columns
    val before_2009_final_political_party = create_inverse_DF("political_party", before_2009)
      .withColumnRenamed("features_idf", "features_idf_before_2009")
    val after_2009_final_political_party = create_inverse_DF("political_party", after_2009)
      .withColumnRenamed("features_idf", "features_idf_after_2009")

    // Join the smallest between two datasets to avoid nulls and compute cosine similarity between the speeches
    // before and after 2009. Then show the results ascending because small similarity means big distance
    println("Most different pairs of party's with respect to speeches are:")
    if (before_2009_final_political_party.rdd.count() >= after_2009_final_political_party.rdd.count()) {
      val results = after_2009_final_political_party
        .join(before_2009_final_political_party, Seq("political_party"))
        .withColumn("Similarity", cosSimilarity(col("features_idf_before_2009"), col("features_idf_after_2009")))
        .sort(col("Similarity").asc)
        .select("political_party", "Similarity")
      results.show()
    }else{
      val results = before_2009_final_political_party
        .join(after_2009_final_political_party, Seq("political_party"))
        .withColumn("Similarity", cosSimilarity(col("features_idf_after_2009"), col("features_idf_before_2009")))
        .sort(col("Similarity").asc)
        .select("political_party", "Similarity")
      results.show()
    }
  }
}
