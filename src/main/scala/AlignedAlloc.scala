import scala.scalanative.unsafe.Ptr
import scala.scalanative.unsafe.*
import scala.scalanative.libc.stdlib
import scala.scalanative.runtime.fromRawPtr
import scala.scalanative.runtime.RawPtr

object AlignedAlloc:

  private val Alignment = 64
  inline def isAligned(ptr: Ptr[?]): Boolean =
    (ptr.toLong & (Alignment - 1)) == 0

  def malloc[A](length: Int)(using Tag[A]): AlignedPtr[A] =
    val elemSize = sizeof[A].toInt
    val bytesNeeded = length * elemSize
    val totalBytes = bytesNeeded + (Alignment - 1)

    // allocate raw block
    val raw = stdlib.malloc(totalBytes.toLong).asInstanceOf[Ptr[Byte]]
    if raw == null then throw new OutOfMemoryError("malloc failed")

    // compute offset to next aligned address
    val addr = raw.toLong
    val misalignment = (addr % Alignment).toInt
    val adjustment =
      if misalignment == 0 then 0
      else Alignment - misalignment

    // use pointer arithmetic (legal!)
    val aligned = (raw + adjustment).asInstanceOf[Ptr[A]]

    AlignedPtr(aligned, raw, length)

  def free[A](ptr: AlignedPtr[A]): Unit =
    stdlib.free(ptr.original)

final case class AlignedPtr[A](
    aligned: Ptr[A],
    original: Ptr[Byte],
    length: Int
)
