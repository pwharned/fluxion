object IrisData:
  def loadFromFile(path: String): Array[(Array[Double], Double)] =
    val lines = scala.io.Source.fromFile(path).getLines().drop(1).toArray
    lines.flatMap { line =>
      val parts = line.split(",")
      val features = Array(parts(0).toDouble, parts(1).toDouble)
      val label = parts(4) match
        case "Iris-setosa"     => Some(0.0)
        case "Iris-versicolor" => Some(1.0)
        case _ => None // skip virginica for binary classification
      label.map(l => (features, l))
    }
