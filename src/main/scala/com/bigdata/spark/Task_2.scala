//package com.bigdata.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object Task_2 {
  def main(args: Array[String]): Unit = {

    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

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
    vectorizedDF.show(25, truncate = false)

    val idf = new IDF()
      .setInputCol("features")
      .setOutputCol("features_idf")
      .fit(vectorizedDF)

    val inverseDF = idf.transform(vectorizedDF)

    inverseDF.printSchema()
    inverseDF.show(10, false)


    val asDense = udf((v: Vector) => v.toDense) //transform to dense matrix

    val vectorsDF = inverseDF.select('features_idf)
      .withColumn("dense_features", asDense($"features_idf")).drop(col("features_idf"))
      .withColumn("id", monotonically_increasing_id())
    vectorsDF.show(11, false)

    vectorsDF.printSchema()


    val cosSimilarity = udf { (x: Vector, y: Vector) =>
      val v1 = x.toArray
      val v2 = y.toArray
      val l1 = scala.math.sqrt(v1.map(x => x*x).sum)
      val l2 = scala.math.sqrt(v2.map(x => x*x).sum)
      val scalar = v1.zip(v2).map(p => p._1*p._2).sum
      scalar/(l1*l2)
    }


    val rows = new VectorAssembler().setInputCols(vectorsDF.columns).setOutputCol("vs")
      .transform(vectorsDF)
      .select("vs")
      .rdd

    val rdd_rows = rows.map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
      .map(org.apache.spark.mllib.linalg.Vectors.fromML)

    rdd_rows.foreach(f => println(f.toDense))

    val mat: RowMatrix = new RowMatrix(rdd_rows)

    // Compute similar columns with estimation using DIMSUM
    val approx = mat.columnSimilarities(.7)
    val approxEntries = approx.entries.map { case MatrixEntry(i, j, v) => ((i, j), v) }

//    approxEntries.foreach(println)
  }
}
