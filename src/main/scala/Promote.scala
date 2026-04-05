trait Promote[A, B]:
  type Out


given Promote[Int, Int] with
  type Out = Int

given Promote[Int, Float] with
  type Out = Float

given Promote[Int, Double] with
  type Out = Double

given Promote[Float, Int] with
  type Out = Float

given Promote[Float, Float] with
  type Out = Float

given Promote[Float, Double] with
  type Out = Double

given Promote[Double, Int] with
  type Out = Double

given Promote[Double, Float] with
  type Out = Double

given Promote[Double, Double] with
  type Out = Double



