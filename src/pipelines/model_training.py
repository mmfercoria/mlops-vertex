# model_training.py

from sklearn.datasets import load_iris
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse
import json

def main(train_path, model_path, metrics_path):
    # Inicializar Spark
    spark = SparkSession.builder.appName("IrisModelTraining").getOrCreate()

    # Cargar el conjunto de datos Iris
    iris_data = load_iris()
    data = iris_data['data']
    target = iris_data['target']

    # Crear un DataFrame de Spark
    columns = iris_data['feature_names'] + ['label']
    data_with_labels = [list(data[i]) + [target[i]] for i in range(len(target))]
    df = spark.createDataFrame(data_with_labels, schema=columns)

    # Convertir características en vector
    assembler = VectorAssembler(inputCols=iris_data['feature_names'], outputCol="features")
    df = assembler.transform(df)

    # Dividir en conjuntos de entrenamiento y prueba
    train_data, test_data = df.randomSplit([0.8, 0.2])

    # Modelo de clasificación
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    model = dt.fit(train_data)

    # Evaluar el modelo
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    # Guardar el modelo
    model.write().overwrite().save(model_path)

    # Guardar las métricas
    metrics = {
        'accuracy': accuracy
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=False, help="Ruta de entrenamiento (no se usa en este caso)")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta para guardar el modelo")
    parser.add_argument("--metrics_path", type=str, required=True, help="Ruta para guardar las métricas")
    args = parser.parse_args()

    main(args.train_path, args.model_path, args.metrics_path)
