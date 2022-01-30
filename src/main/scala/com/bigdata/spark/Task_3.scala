import com.johnsnowlabs.nlp.annotator.{SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.keyword.yake.YakeKeywordExtraction
import com.johnsnowlabs.nlp.base.DocumentAssembler
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.language.postfixOps

object Task_3 {
  def main(args: Array[String]): Unit = {
    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val sc: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate() // Create Spark Session

    import sc.implicits._

    val initial_data = sc // Read the data and create a column Year that contains only the year from the date
      .read
      .option("header", value = true)
      .csv("data_dropna_rm_morestop.csv")
      .withColumn("Year", split(col("sitting_date"), "/")
        .getItem(2))
      .drop(col("sitting_date"))

    def do_preprocess(member_or_party: String): Unit ={

      val cosSimilarity = udf { (x: Vector, y: Vector) => // Udf that calculates cosine Similarity of vectors
        val v1 = x.toArray
        val v2 = y.toArray
        val l1 = scala.math.sqrt(v1.map(x => x * x).sum)
        val l2 = scala.math.sqrt(v2.map(x => x * x).sum)
        val scalar = v1.zip(v2).map(p => p._1 * p._2).sum
        scalar / (l1 * l2)
      }

      val data = initial_data
        .groupBy(member_or_party, "Year")
        .agg(concat_ws(" ", collect_list("speech_processed")) as "speeches")

      var tokenizer = new RegexTokenizer() //Create tokenizer that splits base on a pattern
        .setPattern(" ")
        .setInputCol("speeches")
        .setOutputCol("tokens")

      val tokenized_df = tokenizer //Tokenize data
        .transform(data)

      tokenizer = null

      var vectorizer = new CountVectorizer() //Use vectorizer to convert text to vector
        .setInputCol("tokens")
        .setOutputCol("features")
        .setVocabSize(5000)
        .setMinDF(3)
        .fit(tokenized_df)

      val vectorizedDF = vectorizer
        .transform(tokenized_df)
        .drop("speeches", "tokens")

      vectorizer = null

      var idf = new IDF() // Compute IDF of features from CountVectorizer
        .setInputCol("features")
        .setOutputCol("features_idf")
        .fit(vectorizedDF)

      var inverseDF = idf
        .transform(vectorizedDF) // Transform

      idf = null

      def addColumnIndex(df: DataFrame) = { // function that helps to create increasingly id column
        sc
          .sqlContext
          .createDataFrame(
          df
            .rdd
            .zipWithIndex
            .map {
            case (row, index) => Row
              .fromSeq(row
                .toSeq :+ index)
          },
          // Create schema for index column
          StructType(df
            .schema
            .fields :+ StructField("index", LongType, nullable = false)))
      }

      var inverseDFIndex = addColumnIndex(inverseDF)

      // split dataset in two equal subsets
      var first_part_of_dataset = inverseDFIndex
        .where(col("index") < inverseDF
          .rdd
          .count() / 2)
        .drop("index", "features")
      var second_part_of_dataset = inverseDFIndex
        .where(col("index") >= inverseDF
          .rdd
          .count() / 2)
        .drop("index", "features")

      inverseDF = null
      inverseDFIndex = null

      // add incremental index to rows and rename columns
      val first_part_of_dataset_index = addColumnIndex(first_part_of_dataset)
        .withColumnRenamed(member_or_party, member_or_party+"_A")
        .withColumnRenamed("Year", "Year_A")
        .withColumnRenamed("features_idf", "features_idf_A")
        .withColumnRenamed("index", "index_A")
      val second_part_of_dataset_index = addColumnIndex(second_part_of_dataset)
        .withColumnRenamed(member_or_party, member_or_party+"_B")
        .withColumnRenamed("Year", "Year_B")
        .withColumnRenamed("features_idf", "features_idf_B")
        .withColumnRenamed("index", "index_B")

      first_part_of_dataset = null
      second_part_of_dataset = null

      // calculate cosine Similarity between the two dataset partitions
      var member_names_differences = first_part_of_dataset_index
        .join(second_part_of_dataset_index, first_part_of_dataset_index("index_A") === second_part_of_dataset_index("index_B"), "inner")
        .withColumn("cosSimilarity", cosSimilarity(col("features_idf_A"), col("features_idf_B")))
        .drop("index_A", "index_B")
        .sort(col("cosSimilarity").asc)
        .limit(50)

      var list = member_names_differences // collect distinct member names or party's
        .select(member_or_party+"_A")
        .distinct()
        .rdd
        .collect()
        .toList.union(member_names_differences.select(member_or_party+"_B").distinct().rdd.collect().toList)
        .distinct

      var years_list = member_names_differences // collect distinct years
        .select("Year_A")
        .sort(col("Year_A").asc)
        .distinct()
        .rdd
        .collect()
        .toList.union(member_names_differences.select("Year_B").sort(col("Year_B").asc).distinct().rdd.collect().toList)
        .distinct

      member_names_differences = null

      // repeat for every member or party and every year and find keywords
      list
        .foreach(m=> {
        years_list
          .foreach(y=> {
          println("Keywords for", member_or_party, m(0), "and year", y(0), "are:")
          find_keywords_YakeKeywordExtraction(member_or_party, m(0), "Year", y(0))
          find_keywords_TfIdf(member_or_party, m(0), "Year", y(0))
        })})

      list = null
      years_list = null
    }

    def find_keywords_YakeKeywordExtraction(c: String, value: Any, c1: String, value1: Any): Unit = { // function that finds keywords

      val preprocessed_speeches = initial_data // choose the data
        .where(col(c) === value && col(c1) === value1)

      val documentAssembler = new DocumentAssembler()
        .setInputCol("speech_processed")
        .setOutputCol("document")

      val sentenceDetector = new SentenceDetector() // detect sentences. this is unnecessary because we have
      // preprocessed the data but we let it to be ok in all cases
        .setInputCols("document")
        .setOutputCol("sentence")

      val token = new Tokenizer() // tokenize based on specific characters
        .setInputCols("sentence")
        .setOutputCol("token")
        .setContextChars(Array("(", ")", "?", "!", ".", ",", " "))

      val keywords = new YakeKeywordExtraction() // use YakeKeywordExtraction to extract keywords
        .setInputCols("token")
        .setOutputCol("keywords")
        .setThreshold(0.5f) // set threshold that all keywords will be lower than this
        .setMinNGrams(1) // find keywords with at least one word
        .setNKeywords(10) // find 10 keywords

      val pipeline = new Pipeline() // create pipeline
        .setStages(Array(
        documentAssembler,
        sentenceDetector,
        token,
        keywords
      ))

      var result = pipeline // fit pipeline
        .fit(preprocessed_speeches)
        .transform(preprocessed_speeches)

      // combine the result and score (contained in keywords.metadata)
      var scores = result
        .selectExpr("explode(arrays_zip(keywords.result, keywords.metadata)) as resultTuples")
        .select($"resultTuples.0" as "keyword", $"resultTuples.1.score")

      result = null

      val toDouble = udf[Double, String]( _.toDouble) // function top convert string to double

      val scores_cast = scores // convert scores to Double so we can sort by this column
        .withColumn("score", toDouble(scores("score")))

      scores = null

      // Order ascending, as lower scores means higher importance
      scores_cast
        .groupBy("keyword")
        .sum("score")
        .orderBy("sum(score)")
        .show(10, truncate = false)
    }

    def find_keywords_TfIdf(c: String, value: Any, c1: String, value1: Any): Unit = { // function that finds keywords

      val preprocessed_speeches = initial_data // choose the data
        .where(col(c) === value && col(c1) === value1)
        .select(split(col("speech_processed"), " "))
        .withColumn("doc_id", monotonically_increasing_id())
        .withColumnRenamed("split(speech_processed,  , -1)", "speech_processed")

      val columns = preprocessed_speeches // take all words from preprocessed speeches as tokens
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

    do_preprocess("member_name") // start finding keywords for every member and year
    do_preprocess("political_party") // start finding keywords for every party and year
  }
}
