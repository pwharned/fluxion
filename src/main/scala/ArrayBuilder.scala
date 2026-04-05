// Add this trait (in Numeric.scala or a separate file)
trait ArrayBuilder[A]:
  def newArray(size: Int): Array[A]
object ArrayBuilder:
  given ArrayBuilder[Int] with
    def newArray(size: Int) = new Array[Int](size)
  given ArrayBuilder[Float] with
    def newArray(size: Int) = new Array[Float](size)
  given ArrayBuilder[Double] with
    def newArray(size: Int) = new Array[Double](size)
