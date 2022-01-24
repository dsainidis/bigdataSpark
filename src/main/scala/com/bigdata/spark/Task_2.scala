import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, CountVectorizer, IDF, MinHashLSH, RegexTokenizer}
import org.apache.spark.sql.functions.{col, collect_list, concat_ws}
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object Task_2 {
  def main(args: Array[String]): Unit = {
    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val ss: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("task2")
      .getOrCreate() // Create Spark Session

    // Read input file, select the member_name, the speeches and the political_party and drop nulls
    val inputFile = "./data_dropna_rm_morestop.csv"
    val rawDF = ss
      .read
      .option("header", "true")
      .csv(inputFile)
    val tempDF = rawDF
      .select("member_name", "speech_processed", "political_party")
      .na
      .drop(Seq("member_name"))

    val df = tempDF // group by member name and aggregate their speeches
      .groupBy("member_name", "political_party")
      .agg(concat_ws(" ", collect_list("speech_processed")) as "speeches")

    val tokenizer = new RegexTokenizer() //Create tokenizer that splits base on a pattern
      .setPattern(" ")
      .setInputCol("speeches")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer
      .transform(df)

    val vectorizer = new CountVectorizer() //Use vectorizer to convert text to vector
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(5000)
      .setMinDF(3)
      .fit(tokenized_df)

    val vectorizedDF = vectorizer.transform(tokenized_df).drop("speeches", "tokens") //Vectorize data

    val idf = new IDF() // Compute IDF of rawFeatures from CountVectorizer
      .setInputCol("features")
      .setOutputCol("features_idf")
      .fit(vectorizedDF)

    val inverseDF = idf
      .transform(vectorizedDF) // Transform

    def addColumnIndex(df: DataFrame) = { // function that helps to create increasingly id column
      ss.sqlContext.createDataFrame(
        df.rdd.zipWithIndex.map {
          case (row, index) => Row.fromSeq(row.toSeq :+ index)
        },
        // Create schema for index column
        StructType(df.schema.fields :+ StructField("index", LongType, nullable = false)))
    }

    val inverseDFIndex = addColumnIndex(inverseDF).drop("features")

    var mh = new MinHashLSH()
      .setNumHashTables(100)
      .setInputCol("features_idf")
      .setOutputCol("hashes")

    // the dataset is split into two parts for the function approxSimilarityJoin, otherwise there will
    // be produced duplicate pairs like (4,4).
    val first_part_of_dataset = inverseDFIndex.where(col("index") < inverseDFIndex.rdd.count() / 2)
    val second_part_of_dataset = inverseDFIndex.where(col("index") >= inverseDFIndex.rdd.count() / 2)

    var model_mh = mh.fit(first_part_of_dataset) // fit the BucketedRandomProjectionLSH model

    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model_mh.transform(first_part_of_dataset).show()

    println("Approximately JaccardDistance distance smaller than 0.1:")
    var results_Jaccard = model_mh.approxSimilarityJoin(first_part_of_dataset, second_part_of_dataset, .1, "JaccardDistance")
      .select(col("datasetA.index").alias("idA"),
        col("datasetB.index").alias("idB"),
        col("datasetA.member_name").alias("member_name_A"),
        col("datasetB.member_name").alias("member_name_B"),
        col("datasetA.political_party").alias("political_party_A"),
        col("datasetB.political_party").alias("political_party_B"),
        col("JaccardDistance"))
      .sort(col("JaccardDistance"))

    results_Jaccard.show(results_Jaccard.count().asInstanceOf[Int],truncate = false)

    results_Jaccard = null
    model_mh = null
    mh = null

    var brp = new BucketedRandomProjectionLSH() // We will solve the all-pairs similarity problem using
      // BucketedRandomProjectionLSH
      .setBucketLength(5.0)
      .setNumHashTables(300)
      .setInputCol("features_idf")
      .setOutputCol("hashes")

    var model_brp = brp.fit(first_part_of_dataset)
    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model_brp.transform(first_part_of_dataset).show()

    println("Approximately EuclideanDistance distance smaller than 1:")
    var results_Euclidean = model_brp.approxSimilarityJoin(first_part_of_dataset, second_part_of_dataset, .1, "EuclideanDistance")
      .select(col("datasetA.index").alias("idA"),
        col("datasetB.index").alias("idB"),
        col("datasetA.member_name").alias("member_name_A"),
        col("datasetB.member_name").alias("member_name_B"),
        col("datasetA.political_party").alias("political_party_A"),
        col("datasetB.political_party").alias("political_party_B"),
        col("EuclideanDistance"))
      .sort(col("EuclideanDistance"))

    results_Euclidean.show(results_Euclidean.count().asInstanceOf[Int],truncate = false)

    results_Euclidean = null
    model_brp = null
    brp = null
  }
}
