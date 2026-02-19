object Main:

  def main(args: Array[String]): Unit =
    println("=== Logistic Regression: Iris (Binary, 2 features) ===\n")

    val data = IrisData.loadFromFile("iris.data")

    var w0 = 0.1
    var w1 = 0.1
    var bias = 0.0

    val lr = 0.1

    // ---------- Loss functions: fully expanded, no intermediate vals ----------

    // Loss w.r.t. w0 (everything else is constant)
    inline def lossW0(
        w0Param: Double,
        w1Const: Double,
        biasConst: Double,
        x0: Double,
        x1: Double,
        y: Double
    ): Double =
      // Fully expanded: (sigmoid(z) - y)^2
      // sigmoid = 1/(1 + exp(-z)), z = w0*x0 + w1*x1 + bias
      ((1.0 / (1.0 + math.exp(
        -(w0Param * x0 + w1Const * x1 + biasConst)
      ))) - y) *
        ((1.0 / (1.0 + math.exp(
          -(w0Param * x0 + w1Const * x1 + biasConst)
        ))) - y)

    inline def lossW1(
        w1Param: Double,
        w0Const: Double,
        biasConst: Double,
        x0: Double,
        x1: Double,
        y: Double
    ): Double =
      ((1.0 / (1.0 + math.exp(
        -(w0Const * x0 + w1Param * x1 + biasConst)
      ))) - y) *
        ((1.0 / (1.0 + math.exp(
          -(w0Const * x0 + w1Param * x1 + biasConst)
        ))) - y)

    inline def lossBias(
        biasParam: Double,
        w0Const: Double,
        w1Const: Double,
        x0: Double,
        x1: Double,
        y: Double
    ): Double =
      ((1.0 / (1.0 + math.exp(
        -(w0Const * x0 + w1Const * x1 + biasParam)
      ))) - y) *
        ((1.0 / (1.0 + math.exp(
          -(w0Const * x0 + w1Const * x1 + biasParam)
        ))) - y)

    // ---------- Utility: full loss & accuracy (runtime only) ----------

    def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))

    def computeLoss(): Double =
      var sum = 0.0
      for (x, y) <- data do
        val pred = sigmoid(w0 * x(0) + w1 * x(1) + bias)
        val e = pred - y
        sum += e * e
      sum / data.length

    def accuracy(): Double =
      var correct = 0
      for (x, y) <- data do
        val pred =
          if sigmoid(w0 * x(0) + w1 * x(1) + bias) > 0.5 then 1.0 else 0.0
        if pred == y then correct += 1
      correct.toDouble / data.length * 100.0

    println(
      f"Initial: loss=${computeLoss()}%.4f, accuracy=${accuracy()}%.2f%%\n"
    )

    // ---------- Training loop ----------

    for epoch <- 0 until 100 do
      var g0 = 0.0
      var g1 = 0.0
      var gb = 0.0

      for (x, y) <- data do
        val x0 = x(0)
        val x1 = x(1)
        diffFn((x: Double) => scala.math.exp(x))

        // Create gradient functions via compile-time AD
        val g0Fn = diffFn((w: Double) => lossW0(w, w1, bias, x0, x1, y))
        val g1Fn = diffFn((w: Double) => lossW1(w, w0, bias, x0, x1, y))
        val gbFn = diffFn((b: Double) => lossBias(b, w0, w1, x0, x1, y))

        g0 += g0Fn(w0)
        g1 += g1Fn(w1)
        gb += gbFn(bias)

      g0 /= data.length
      g1 /= data.length
      gb /= data.length

      w0 -= lr * g0
      w1 -= lr * g1
      bias -= lr * gb

      if epoch % 20 == 0 then
        println(
          f"Epoch $epoch%3d: loss=${computeLoss()}%.4f, accuracy=${accuracy()}%.2f%%"
        )

    println(f"\nFinal: loss=${computeLoss()}%.4f, accuracy=${accuracy()}%.2f%%")
    println(f"Learned weights: w0=$w0%.4f, w1=$w1%.4f, bias=$bias%.4f")
