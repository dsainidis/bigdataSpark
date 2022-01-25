package org.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{LongType, StructField, StructType}

object Task_6 {
  def main(args: Array[String]): Unit = {

    Logger
      .getLogger("org")
      .setLevel(Level
        .ERROR) // Hide logger from console if it is not an ERROR

    val ss: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("task6")
      .getOrCreate() // Create Spark Session

    def addColumnIndex(df: DataFrame) = { // function that helps to create increasingly id column
      ss.sqlContext.createDataFrame(
        df.rdd.zipWithIndex.map {
          case (row, index) => Row.fromSeq(row.toSeq :+ index)
        },
        // Create schema for index column
        StructType(df.schema.fields :+ StructField("index", LongType, nullable = false)))
    }

    // Read input file, select the speeches and the date and create a column that contain only the years as we dont need
    // the rest date
    val inputFile = "./data_dropna_rm_morestop.csv"
    val inputDF = ss // read the data and drop nulls
      .read
      .option("header", "true")
      .csv(inputFile)
      .na
      .drop()

    val inputDFIndex = addColumnIndex(inputDF) // add monotonically increasing id to the dataframe

    // Encode every categorical feature of the data using StringIndexer
    val target_categories = new StringIndexer() // keep separate the target categories
      .setInputCol("political_party")
      .setOutputCol("indexed")
      .fit(inputDFIndex)
      .transform(inputDFIndex)
      .select("political_party", "indexed")

    val tmp = new StringIndexer()
      .setInputCol("member_name")
      .setOutputCol("member_name_encoded")
      .fit(inputDFIndex)
      .transform(inputDFIndex)
      .drop("member_name", "speech")

    val tmp_1 = new StringIndexer()
      .setInputCol("sitting_date")
      .setOutputCol("sitting_date_encoded")
      .fit(tmp)
      .transform(tmp)
      .drop("sitting_date")

    val tmp_2 = new StringIndexer()
      .setInputCol("parliamentary_period")
      .setOutputCol("parliamentary_period_encoded")
      .fit(tmp_1)
      .transform(tmp_1)
      .drop("parliamentary_period")

    val tmp_3 = new StringIndexer()
      .setInputCol("parliamentary_session")
      .setOutputCol("parliamentary_session_encoded")
      .fit(tmp_2)
      .transform(tmp_2)
      .drop("parliamentary_session")

    val tmp_4 = new StringIndexer()
      .setInputCol("parliamentary_sitting")
      .setOutputCol("parliamentary_sitting_encoded")
      .fit(tmp_3)
      .transform(tmp_3)
      .drop("parliamentary_sitting")

    val tmp_5 = new StringIndexer()
      .setInputCol("political_party")
      .setOutputCol("political_party_encoded")
      .fit(tmp_4)
      .transform(tmp_4)
      .drop("political_party")

    val tmp_6 = new StringIndexer()
      .setInputCol("government")
      .setOutputCol("government_encoded")
      .fit(tmp_5)
      .transform(tmp_5)
      .drop("government")

    val tmp_7 = new StringIndexer()
      .setInputCol("member_region")
      .setOutputCol("member_region_encoded")
      .fit(tmp_6)
      .transform(tmp_6)
      .drop("member_region")

    val tmp_8 = new StringIndexer()
      .setInputCol("roles")
      .setOutputCol("roles_encoded")
      .fit(tmp_7)
      .transform(tmp_7)
      .drop("roles")

    val tmp_9 = new StringIndexer()
      .setInputCol("member_gender")
      .setOutputCol("member_gender_encoded")
      .fit(tmp_8)
      .transform(tmp_8)
      .drop("member_gender")

    val tokenizer = new RegexTokenizer() //Tokenizer
      .setPattern(" ")
      .setInputCol("speech_processed")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer
      .transform(tmp_9)
      .drop("speech_processed")

    val vectorizer = new CountVectorizer() //CountVectorizer
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(5000)
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
      .drop("tokens", "features")

    val feature_columns = inverseDF
      .drop("_c0")
      .columns

    val assembler = new VectorAssembler() // put all the features together
      .setInputCols(feature_columns)
      .setOutputCol("features")

    val data = assembler
      .transform(inverseDF)
      .withColumnRenamed("political_party_encoded", "label") // Rename target column to label
    // because RandomForestClassifier demands it

    val splitSeed = 5043
    val Array(trainingData, testData) = data // split the data to training and test data
      .randomSplit(Array(0.8, 0.2), splitSeed)

    val rf = new RandomForestClassifier() // create random forest Classifier
      .setLabelCol("label")
      .setMaxBins(5000)
      .setFeaturesCol("features")

    val trainedModel = rf // train model
      .fit(trainingData)

    val predictions = trainedModel // predict
      .transform(testData)
    val target_categories_distinct = target_categories // take the distinct target categories and their numerical
      // representations
      .select("political_party", "indexed")
      .distinct()
    val final_DF = predictions // take predictions and the political party that each prediction corresponds
      .join(target_categories_distinct, predictions("prediction") === target_categories_distinct("indexed"), "inner")

    println("Predictions and political party's that corresponds are:")
    final_DF.show() // print
    println("Predictions with all other information are:")
    final_DF.join(inputDFIndex, final_DF("index") === inputDFIndex("index"), "inner").show() // for the predictions
    // show and the rest of information from the initial dataframe

    println(trainedModel // print accuracy of model
      .evaluate(testData)
      .accuracy)
  }
}
