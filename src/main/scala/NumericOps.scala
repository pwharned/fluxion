trait NumericOps[A]:
  def zero: A
  def one: A
  def add(x: A, y: A): A
  def sub(x: A, y: A): A
  def mul(x: A, y: A): A
  def div(x: A, y: A): A
  def neg(x: A): A
  def fromInt(i: Int): A
  def fromDouble(i: Double): A
  def toDouble(x: A): Double
  // Math functions
  def exp(x: A): A
  def log(x: A): A
  def sqrt(x: A): A
object NumericOps:
  given NumericOps[Int] with
    def zero = 0
    def one = 1
    def fromDouble(i: Double): Int = i.toInt
    def add(x: Int, y: Int) = x + y
    def sub(x: Int, y: Int) = x - y
    def mul(x: Int, y: Int) = x * y
    def div(x: Int, y: Int) = x / y
    def neg(x: Int) = -x
    def fromInt(i: Int) = i
    def toDouble(x: Int) = x.toDouble
    def exp(x: Int) = math.exp(x.toDouble).toInt
    def log(x: Int) = math.log(x.toDouble).toInt
    def sqrt(x: Int) = math.sqrt(x.toDouble).toInt
  given NumericOps[Float] with
    def zero = 0.0f
    def one = 1.0f
    def fromDouble(i: Double): Float = i.toFloat
    def add(x: Float, y: Float) = x + y
    def sub(x: Float, y: Float) = x - y
    def mul(x: Float, y: Float) = x * y
    def div(x: Float, y: Float) = x / y
    def neg(x: Float) = -x
    def fromInt(i: Int) = i.toFloat
    def toDouble(x: Float) = x.toDouble
    def exp(x: Float) = math.exp(x.toDouble).toFloat
    def log(x: Float) = math.log(x.toDouble).toFloat
    def sqrt(x: Float) = math.sqrt(x.toDouble).toFloat
  given NumericOps[Double] with
    def fromDouble(i: Double): Double = i
    def zero = 0.0
    def one = 1.0
    def add(x: Double, y: Double) = x + y
    def sub(x: Double, y: Double) = x - y
    def mul(x: Double, y: Double) = x * y
    def div(x: Double, y: Double) = x / y
    def neg(x: Double) = -x
    def fromInt(i: Int) = i.toDouble
    def toDouble(x: Double) = x
    def exp(x: Double) = math.exp(x)
    def log(x: Double) = math.log(x)
    def sqrt(x: Double) = math.sqrt(x)
