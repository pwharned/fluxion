object Main:
  def main(args: Array[String]): Unit =
    println(
      "=== Logistic Regression: Iris (Binary, 2 features) with Tensors ===\n"
    )
    // Load data
    val rawData = IrisData.loadFromFile("iris.data")
    // Convert to tensors
    val features = rawData.map(_._1)
    val labels = rawData.map(_._2)
    val X = Tensor.fromArray(features.flatten, Array(features.length, 2))
    val y = Tensor.fromArray(labels, Array(labels.length))
    println(s"Loaded ${X.shape(0)} samples with ${X.shape(1)} features")
    println(s"X shape: [${X.shape.mkString(", ")}]")
    println(s"y shape: [${y.shape.mkString(", ")}]")
    // Debug: Check data
    println(s"\nFirst 5 samples:")
    for i <- 0 until 5 do
      val xi = X.row(i)
      println(f"  x[$i] = [${xi(0)}%.2f, ${xi(1)}%.2f], y[$i] = ${y(i)}%.0f")
    println(s"\nLabel distribution:")
    val labels0 = labels.count(_ == 0.0)
    val labels1 = labels.count(_ == 1.0)
    println(s"  Class 0: $labels0")
    println(s"  Class 1: $labels1")
    // Initialize parameters
    var weights = Tensor.fromArray(Array(0.1, 0.1))
    var bias = 0.0
    val lr = 0.1
    val nSamples = X.shape(0)
    // ========== Helper Functions ==========
    def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))
    def forward(): Tensor[Double] =
      val predictions = Tensor.zeros[Double](Array(nSamples))
      var i = 0
      while i < nSamples do
        val xi = X.row(i)
        val z = xi.dot(weights) + bias
        predictions.update(i)(sigmoid(z))
        i += 1
      predictions
    def computeLoss(): Double =
      val pred = forward()
      val diff = pred - y
      (diff.square).mean
    def accuracy(): Double =
      val pred = forward()
      var correct = 0
      var i = 0
      while i < nSamples do
        val predicted = if pred(i) > 0.5 then 1.0 else 0.0
        if predicted == y(i) then correct += 1
        i += 1
      correct.toDouble / nSamples * 100.0
    // Debug: Check initial predictions
    println(s"\nInitial predictions (first 5):")
    val initPred = forward()
    for i <- 0 until 5 do
      println(f"  pred[$i] = ${initPred(i)}%.4f (expected ${y(i)}%.0f)")
    println(
      f"\nInitial weights: [${weights(0)}%.4f, ${weights(1)}%.4f], bias=$bias%.4f"
    )
    println(
      f"Initial: loss=${computeLoss()}%.4f, accuracy=${accuracy()}%.2f%%\n"
    )
    // ========== Sanity check before training ==========
    println("=== Sanity check: manual prediction ===")
    val testX = X.row(0)
    val testY = y(0)
    println(f"Sample 0: x=[${testX(0)}%.2f, ${testX(1)}%.2f], y=$testY%.0f")
    val testZ = testX.dot(weights) + bias
    val testPred = sigmoid(testZ)
    println(
      f"z = ${testX(0)}%.2f * ${weights(0)}%.2f + ${testX(1)}%.2f * ${weights(1)}%.2f + $bias%.2f = $testZ%.4f"
    )
    println(f"sigmoid($testZ%.4f) = $testPred%.4f")
    println(f"error = $testPred%.4f - $testY%.0f = ${testPred - testY}%.4f")
    println()
    // ========== Training Loop ==========
    var start = System.nanoTime()
    for epoch <- 0 until 100 do
      var gradW = Tensor.zeros[Double](Array(2))
      var gradB = 0.0
      // Compute gradients
      var i = 0
      while i < nSamples do
        val xi = X.row(i)
        val yi = y(i)
        // Forward pass
        val z = xi.dot(weights) + bias
        val pred = sigmoid(z)
        // Backward pass
        val error = pred - yi
        val sigmoidDeriv = pred * (1.0 - pred)
        val dLossDz = 2.0 * error * sigmoidDeriv
        // Accumulate gradients
        var j = 0
        while j < 2 do
          gradW.update(j)(gradW(j) + dLossDz * xi(j))
          j += 1
        gradB += dLossDz
        i += 1
      // Average gradients
      gradW = gradW / nSamples.toDouble
      gradB = gradB / nSamples
      // Debug first epoch
      if epoch == 0 then
        println("=== First epoch gradient check ===")
        println(f"gradW = [${gradW(0)}%.6f, ${gradW(1)}%.6f]")
        println(f"gradB = $gradB%.6f")
        println(f"update: weights -= lr * gradW = [${weights(0)}%.4f, ${weights(
            1
          )}%.4f] - $lr * [${gradW(0)}%.4f, ${gradW(1)}%.4f]")
        println()
      // Update parameters
      weights = weights - (gradW * lr)
      bias = bias - (lr * gradB)
      if epoch == 1 then
        println(
          f"After 1 epoch: weights=[${weights(0)}%.4f, ${weights(1)}%.4f], bias=$bias%.4f"
        )
        println(f"Loss after 1 epoch: ${computeLoss()}%.4f")
        println()
      if epoch % 20 == 0 then
        println(
          f"Epoch $epoch%3d: loss=${computeLoss()}%.4f, accuracy=${accuracy()}%.2f%%, " +
            f"w=[${weights(0)}%.4f, ${weights(1)}%.4f], b=$bias%.4f"
        )
    var end = System.nanoTime()
    val trainTime = (end - start) / 1_000_000.0
    println(f"\nTrain time: $trainTime%.2f ms")
    println(f"Final: loss=${computeLoss()}%.4f, accuracy=${accuracy()}%.2f%%")
    println(
      f"Learned weights: [${weights(0)}%.4f, ${weights(1)}%.4f], bias=$bias%.4f"
    )
    // ========== Final predictions check ==========
    println(s"\nFinal predictions (first 10):")
    val finalPred = forward()
    for i <- 0 until 10 do
      val pred = finalPred(i)
      val predClass = if pred > 0.5 then 1.0 else 0.0
      val correct = if predClass == y(i) then "✓" else "✗"
      println(
        f"  pred[$i] = $pred%.4f → class $predClass%.0f (actual ${y(i)}%.0f) $correct"
      )
    // ========== Inference Benchmark ==========
    println("\n=== Inference Benchmark ===")
    val nIterations = 10000
    val testX2 = Tensor.fromArray(Array(5.1, 3.5))
    // Warmup
    for _ <- 0 until 100 do
      val z = testX2.dot(weights) + bias
      sigmoid(z)
    start = System.nanoTime()
    for _ <- 0 until nIterations do
      val z = testX2.dot(weights) + bias
      sigmoid(z)
    end = System.nanoTime()
    val inferenceTimeUs = (end - start) / nIterations.toDouble / 1000.0
    println(f"Average inference time: $inferenceTimeUs%.2f μs per prediction")
    println(
      f"Total for $nIterations predictions: ${(end - start) / 1e6}%.2f ms"
    )
