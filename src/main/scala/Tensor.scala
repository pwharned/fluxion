import scala.scalanative.unsafe.Ptr
import scala.scalanative.unsafe.*
import scala.scalanative.libc.stdlib
import scala.scalanative.runtime.fromRawPtr
import scala.scalanative.runtime.RawPtr
import scala.math.Fractional.Implicits.infixFractionalOps
import scala.math.Integral.Implicits.infixIntegralOps
import scala.math.Numeric.Implicits.infixNumericOps

final class Tensor[A <: Numeric] private (
    val shape: Array[Int],
    private val stride: Array[Int],
    private val offset: Int,
    private val data: AlignedPtr[A]
)(using val ops: NumericOps[A], val tag: Tag[A], builder: ArrayBuilder[A]):
  import scala.scalanative.unsafe.*
  def to[B <: Numeric: Tag: NumericOps: ArrayBuilder]: Tensor[B] =
    val opsB = summon[NumericOps[B]]
    map(a => opsB.fromDouble(ops.toDouble(a)))
  // ========== Indexing ==========
  /** Compute flat index from multi-dimensional indices */
  private def computeIndex(indices: Int*): Int =
    require(
      indices.length == shape.length,
      s"Expected ${shape.length} indices, got ${indices.length}"
    )
    var idx = offset
    var i = 0
    while i < indices.length do
      require(
        indices(i) >= 0 && indices(i) < shape(i),
        s"Index ${indices(i)} out of bounds for dimension $i (size ${shape(i)})"
      )
      idx += indices(i) * stride(i)
      i += 1
    idx

  def apply(indices: Int*): A =
    val idx = computeIndex(indices*) // Changed from indices: _*
    data.aligned(idx)

  /** Set element at indices */
  def update(indices: Int*)(value: A): Unit =
    val idx = computeIndex(indices*) // Changed from indices: _*
    data.aligned(idx) = value

  /** Extract a row from a 2D tensor (creates a copy) */
  def row(rowIdx: Int): Tensor[A] =
    require(ndim == 2, s"row() requires 2D tensor, got ${ndim}D")
    require(
      rowIdx >= 0 && rowIdx < shape(0),
      s"Row index $rowIdx out of bounds for dimension 0 (size ${shape(0)})"
    )
    val nCols = shape(1)
    val result = Tensor.zeros[A](Array(nCols))
    var j = 0
    while j < nCols do
      val value = this.apply(rowIdx, j) // Explicit apply call
      result.update(j)(value) // Explicit update call
      j += 1
    result

  /** Get element from flat index (useful for iteration) */
  private def getFlat(idx: Int): A = data.aligned(offset + idx)

  /** Set element at flat index */
  private def setFlat(idx: Int, value: A): Unit =
    data.aligned(offset + idx) = value
  // ========== Properties ==========
  /** Total number of elements */
  def length: Int = shape.product

  /** Number of dimensions */
  def ndim: Int = shape.length
  // ========== Higher-order Operations ==========
  /** Apply function to each element */
  def map[B <: Numeric: Tag: NumericOps: ArrayBuilder](f: A => B): Tensor[B] =
    val result = Tensor.zeros[B](shape)
    var i = 0
    val len = length
    while i < len do
      result.setFlat(i, f(getFlat(i)))
      i += 1
    result

  /** Combine two tensors element-wise */
  def zipWith[B <: Numeric, C <: Numeric: Tag: NumericOps: ArrayBuilder](
      other: Tensor[B]
  )(
      f: (A, B) => C
  ): Tensor[C] =
    require(
      shape.sameElements(other.shape),
      s"Shape mismatch: ${shape.mkString(",")} vs ${other.shape.mkString(",")}"
    )
    val result = Tensor.zeros[C](shape)
    var i = 0
    val len = length
    while i < len do
      result.setFlat(i, f(getFlat(i), other.getFlat(i)))
      i += 1
    result
  // ========== Binary Tensor Operations (Same Type) ==========
  def +(other: Tensor[A]): Tensor[A] =
    zipWith(other)(ops.add)
  def -(other: Tensor[A]): Tensor[A] =
    zipWith(other)(ops.sub)
  def *(other: Tensor[A]): Tensor[A] =
    zipWith(other)(ops.mul)
  def /(other: Tensor[A]): Tensor[A] =
    zipWith(other)(ops.div)
  // ========== Binary Tensor Operations (Mixed Type with Promotion) ==========
  def +(scalar: A): Tensor[A] = map(ops.add(_, scalar))
  def -(scalar: A): Tensor[A] = map(ops.sub(_, scalar))
  def *(scalar: A): Tensor[A] = map(ops.mul(_, scalar))
  def /(scalar: A): Tensor[A] = map(ops.div(_, scalar))
  // ========== Unary Operations ==========
  def unary_- : Tensor[A] = map(ops.neg)
  // ========== Reductions ==========
  def sum: A =
    var result = ops.zero
    var i = 0
    val len = length
    while i < len do
      result = ops.add(result, getFlat(i))
      i += 1
    result
  def mean: A =
    ops.div(sum, ops.fromInt(length))
  // ========== Linear Algebra ==========
  /** Dot product (only for 1D tensors) */
  def dot(other: Tensor[A]): A =
    require(ndim == 1 && other.ndim == 1, "dot product requires 1D tensors")
    require(
      length == other.length,
      s"Length mismatch: $length vs ${other.length}"
    )
    var result = ops.zero
    var i = 0
    val len = length
    while i < len do
      result = ops.add(result, ops.mul(getFlat(i), other.getFlat(i)))
      i += 1
    result
  // ========== Math Functions ==========
  def exp: Tensor[A] = map(ops.exp)
  def log: Tensor[A] = map(ops.log)
  def sqrt: Tensor[A] = map(ops.sqrt)
  def square: Tensor[A] = map(x => ops.mul(x, x))
  def sigmoid: Tensor[A] =
    map(x => ops.div(ops.one, ops.add(ops.one, ops.exp(ops.neg(x)))))
  // ========== Utilities ==========
  def toArray: Array[A] =
    val result = builder.newArray(length)
    var i = 0
    val len = length
    while i < len do
      result(i) = getFlat(i)
      i += 1
    result
  override def toString: String =
    val preview = toArray.take(10).mkString(", ")
    val suffix = if length > 10 then "..." else ""
    s"Tensor(shape=[${shape.mkString(", ")}], data=[$preview$suffix])"
end Tensor
object Tensor:
  def zeros[A <: Numeric: Tag: NumericOps](shape: Array[Int])(using
      builder: ArrayBuilder[A]
  ): Tensor[A] =
    val ops = summon[NumericOps[A]]
    val length = shape.product
    val data = AlignedAlloc.malloc[A](length)
    val stride = computeRowMajorStride(shape)
    // Initialize to zero
    var i = 0
    while i < length do
      data.aligned(i) = ops.zero
      i += 1
    new Tensor(shape.clone(), stride, 0, data)
  def ones[A <: Numeric: Tag: NumericOps](shape: Array[Int])(using
      builder: ArrayBuilder[A]
  ): Tensor[A] =
    val ops = summon[NumericOps[A]]
    val t = zeros[A](shape)
    var i = 0
    val len = t.length
    while i < len do
      t.setFlat(i, ops.one)
      i += 1
    t
  def fill[A <: Numeric: Tag: NumericOps](shape: Array[Int])(
      value: A
  )(using builder: ArrayBuilder[A]): Tensor[A] =
    val t = zeros[A](shape)
    var i = 0
    val len = t.length
    while i < len do
      t.setFlat(i, value)
      i += 1
    t
  def fromArray[A <: Numeric: Tag: NumericOps: ArrayBuilder](
      arr: Array[A],
      shape: Array[Int]
  ): Tensor[A] =
    require(
      arr.length == shape.product,
      s"Array length ${arr.length} doesn't match shape ${shape.mkString("×")}"
    )
    val t = zeros[A](shape)
    var i = 0
    val len = arr.length
    while i < len do
      t.setFlat(i, arr(i))
      i += 1
    t
  def fromArray[A <: Numeric: Tag: NumericOps: ArrayBuilder](
      arr: Array[A]
  ): Tensor[A] =
    fromArray(arr, Array(arr.length))
  private def computeRowMajorStride(shape: Array[Int]): Array[Int] =
    val n = shape.length
    val stride = new Array[Int](n)
    var acc = 1
    var i = n - 1
    while i >= 0 do
      stride(i) = acc
      acc *= shape(i)
      i -= 1
    stride
  def apply[A <: Numeric: Tag: NumericOps: ArrayBuilder](
      data: Ptr[A],
      shape: Array[Int]
  ): Tensor[A] =
    if !AlignedAlloc.isAligned(data) then
      throw new IllegalArgumentException(
        s"Pointer $data is not 64-byte aligned"
      )
    val length = shape.product
    val alignedPtr = AlignedPtr(
      aligned = data,
      original = data.asInstanceOf[Ptr[Byte]],
      length = length
    )
    val stride = computeRowMajorStride(shape)
    new Tensor(shape.clone(), stride, 0, alignedPtr)
  given intToFloat: Conversion[Tensor[Int], Tensor[Float]] with
    def apply(t: Tensor[Int]): Tensor[Float] = t.to[Float]
  given intToDouble: Conversion[Tensor[Int], Tensor[Double]] with
    def apply(t: Tensor[Int]): Tensor[Double] = t.to[Double]
  given floatToDouble: Conversion[Tensor[Float], Tensor[Double]] with
    def apply(t: Tensor[Float]): Tensor[Double] = t.to[Double]

end Tensor
