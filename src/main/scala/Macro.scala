import scala.quoted._

enum Expr:
  case Const(v: Double)
  case Var(name: String)
  case Add(a: Expr, b: Expr)
  case Sub(a: Expr, b: Expr)
  case Mul(a: Expr, b: Expr)
  case Div(a: Expr, b: Expr)
  case Exp(a: Expr)

import Expr.*

// Pretty-print (for debugging / demo)
def show(e: Expr): String = e match
  case Const(v)   => v.toString
  case Var(n)     => n
  case Add(a, b)  => s"(${show(a)} + ${show(b)})"
  case Sub(a, b)  => s"(${show(a)} - ${show(b)})"
  case Mul(a, b)  => s"(${show(a)} * ${show(b)})"
  case Div(a, b) => s"(${show(a)} / ${show(b)})"
  case Exp(a)    => s"exp(${show(a)})"
// ===== 2. Symbolic differentiation (w.r.t. a single variable) =====

def diff(e: Expr, wrt: String): Expr = e match
  case Const(_)   => Const(0.0)
  case Div(a,b)  => Div(
      Sub(Mul(diff(a, wrt), b), Mul(a, diff(b, wrt))),
      Mul(b, b)
    )
  case Var(n)     => if n == wrt then Const(1.0) else Const(0.0)
  case Add(a, b)  => Add(diff(a, wrt), diff(b, wrt))
  case Sub(a, b)  => Sub(diff(a, wrt), diff(b, wrt))
  case Mul(a, b)  =>
    // product rule: (a*b)' = a'*b + a*b'
    Add(
      Mul(diff(a, wrt), b),
      Mul(a, diff(b, wrt))
    )
  case Exp(a) =>
    simplify(a) match
      case Const(0.0) => Const(1.0)  // e^0 = 1
      case sa => Exp(sa)
// Optional: tiny simplifier (constant folding + trivial zeros/ones)
def simplify(e: Expr): Expr = e match
  case Add(Const(0.0), b) => simplify(b)
  case Add(a, Const(0.0)) => simplify(a)
  case Sub(a, Const(0.0)) => simplify(a)
  case Mul(Const(0.0), _) => Const(0.0)
  case Mul(_, Const(0.0)) => Const(0.0)
  case Mul(Const(1.0), b) => simplify(b)
  case Mul(a, Const(1.0)) => simplify(a)
  case Add(a, b) =>
    (simplify(a), simplify(b)) match
      case (Const(x), Const(y)) => Const(x + y)
      case (sa, sb)             => Add(sa, sb)
  case Sub(a, b) =>
    (simplify(a), simplify(b)) match
      case (Const(x), Const(y)) => Const(x - y)
      case (sa, sb)             => Sub(sa, sb)
  case Mul(a, b) =>
    (simplify(a), simplify(b)) match
      case (Const(x), Const(y)) => Const(x * y)
      case (sa, sb)             => Mul(sa, sb)
  case other => other

// ===== 3. Evaluation of Expr at runtime (for sanity checks) =====

def eval(e: Expr, xVal: Double, varName: String = "x"): Double = e match
  case Const(v)   => v
  case Var(n)     => if n == varName then xVal else
                       throw new IllegalArgumentException(s"Unknown var: $n")
  case Add(a, b)  => eval(a, xVal, varName) + eval(b, xVal, varName)
  case Sub(a, b)  => eval(a, xVal, varName) - eval(b, xVal, varName)
  case Mul(a, b)  => eval(a, xVal, varName) * eval(b, xVal, varName)
  case Div(a, b) => eval(a, xVal, varName) / eval(b, xVal, varName)
  case Exp(a)    => math.exp(eval(a, xVal, varName))
// ===== 4. Macros: from Scala function -> Expr -> derivative -> Scala function =====



inline def diffFn(inline f: Double => Double): Double => Double =
  ${ diffFnImpl('f) }


private def diffFnImpl(f: scala.quoted.Expr[Double => Double])(using Quotes): scala.quoted.Expr[Double => Double] =
  import quotes.reflect.*
  
  // Extract body and convert to symbolic Expr
  def toExpr(func: scala.quoted.Expr[Double => Double]): Expr =
    func match
      // Base case: identity function is the variable
      case '{ (x: Double) => x } => 
        Var("x")
      case '{ (x: Double) => ($a(x): Double) / ($b(x): Double) } =>
        Div(toExpr(a), toExpr(b))
      
      // exp() function call
      case '{ (x: Double) => scala.math.exp($a(x)) } =>
        Exp(toExpr(a))
      // Addition
      case '{ (x: Double) => ($a(x): Double) + ($b(x): Double) } =>
        Add(toExpr(a), toExpr(b))
      
      // Subtraction  
      case '{ (x: Double) => ($a(x): Double) - ($b(x): Double) } =>
        Sub(toExpr(a), toExpr(b))
      
      // Multiplication
      case '{ (x: Double) => ($a(x): Double) * ($b(x): Double) } =>
        Mul(toExpr(a), toExpr(b))
      case '{ (x: Double) => ${scala.quoted.Expr(d: Double)}: Double } =>
        Const(d)

      case _ =>
        report.errorAndAbort(
          s"Unsupported expression\n ${func.show}\n" +
          s"Only +, -, * and constants are currently supported."
        )
  
  // Convert symbolic Expr back to function
  def fromExpr(e: Expr): scala.quoted.Expr[Double => Double] =
    e match
      case Const(v) => 
        '{ (x: Double) => ${scala.quoted.Expr(v)} }
      case Var(_) => 
        '{ (x: Double) => x }
      case Add(a, b) => 
        '{ (x: Double) => ${fromExpr(a)}(x) + ${fromExpr(b)}(x) }
      case Sub(a, b) => 
        '{ (x: Double) => ${fromExpr(a)}(x) - ${fromExpr(b)}(x) }
      case Mul(a, b) => 
        '{ (x: Double) => ${fromExpr(a)}(x) * ${fromExpr(b)}(x) }
  
  val symbolicExpr = toExpr(f)
  val derivative = simplify(diff(symbolicExpr, "x"))
  fromExpr(derivative)

