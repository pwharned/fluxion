import scala.quoted._

enum Expr:
  case Const(v: Double)
  case ConstSymbol(name: String)
  case Var(name: String)
  case Add(a: Expr, b: Expr)
  case Sub(a: Expr, b: Expr)
  case Mul(a: Expr, b: Expr)
  case Div(a: Expr, b: Expr)
  case Exp(a: Expr)

import Expr.*

def show(e: Expr): String = e match
  case Const(v)   => v.toString
  case ConstSymbol(name) => name
  case Var(n)     => n
  case Add(a, b)  => s"(${show(a)} + ${show(b)})"
  case Sub(a, b)  => s"(${show(a)} - ${show(b)})"
  case Mul(a, b)  => s"(${show(a)} * ${show(b)})"
  case Div(a, b) => s"(${show(a)} / ${show(b)})"
  case Exp(a)    => s"exp(${show(a)})"

def diff(e: Expr, wrt: String): Expr = e match
  case Const(_)   => Const(0.0)
  case ConstSymbol(name) => Const(0.0)
  case Div(a,b)  => Div(
      Sub(Mul(diff(a, wrt), b), Mul(a, diff(b, wrt))),
      Mul(b, b)
    )
  case Var(n)     => if n == wrt then Const(1.0) else Const(0.0)
  case Add(a, b)  => Add(diff(a, wrt), diff(b, wrt))
  case Sub(a, b)  => Sub(diff(a, wrt), diff(b, wrt))
  case Mul(a, b)  =>
    Add(
      Mul(diff(a, wrt), b),
      Mul(a, diff(b, wrt))
    )
  case Exp(a) =>
    simplify(a) match
      case sa =>  Mul(Exp(sa), diff(sa, wrt))
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


def eval(e: Expr, xVal: Double, varName: String = "x"): Double = e match
  case Const(v)   => v
  case Var(n)     => if n == varName then xVal else
                       throw new IllegalArgumentException(s"Unknown var: $n")
  case Add(a, b)  => eval(a, xVal, varName) + eval(b, xVal, varName)
  case Sub(a, b)  => eval(a, xVal, varName) - eval(b, xVal, varName)
  case Mul(a, b)  => eval(a, xVal, varName) * eval(b, xVal, varName)
  case Div(a, b) => eval(a, xVal, varName) / eval(b, xVal, varName)
  case Exp(a)    => math.exp(eval(a, xVal, varName))


inline def diffFn(inline f: Double => Double): Double => Double =
  ${ diffFnImpl('f) }

private def diffFnImpl(f: scala.quoted.Expr[Double => Double])(using Quotes): scala.quoted.Expr[Double => Double] =
  import quotes.reflect.*

  val capturedVars = scala.collection.mutable.Map[String, scala.quoted.Expr[Double]]()
  
  val proxyToOriginal = scala.collection.mutable.Map[String, Term]()

  val (paramSym, bodyTerm): (Symbol, Term) =
    f.asTerm match
      case Inlined(_, _, Lambda(List(param:  ValDef), body)) =>
        (param.symbol, body)
      case Lambda(List(param: ValDef), body) =>
        (param.symbol, body)
      case other =>
        report.errorAndAbort(s"Expected a simple lambda Double => Double, got: ${other.show}")

  def extractProxyMappings(term: Term): Unit =
    term match
      case Inlined(_, bindings, expansion) =>
        bindings.foreach {
          case ValDef(proxyName, _, Some(rhs)) =>
            proxyToOriginal(proxyName) = rhs
          case _ =>
        }
        extractProxyMappings(expansion)
      case Block(stats, expr) =>
        stats.foreach {
          case t: Term => extractProxyMappings(t)
          case _ => // Other statements (definitions, etc.)
        }
      case Apply(fun, args) =>
        extractProxyMappings(fun)
        args.foreach(extractProxyMappings)
      case Select(qual, _) =>
        extractProxyMappings(qual)
      case Typed(expr, _) =>
        extractProxyMappings(expr)
      case _ => // Leaf nodes, no further traversal
  
  extractProxyMappings(bodyTerm)

  def normalize(term: Term): Term = term match
    case Inlined(_, _, t) =>
      normalize(t)
    case Block(stats, expr) =>
      normalize(expr)
    case Typed(t, _) =>
      normalize(t)
    case other =>
      other

  def toExprTerm(term: Term): Expr =
    normalize(term) match
      case id @ Ident(_) if id.symbol == paramSym => Var("x")
      case Literal(DoubleConstant(v)) =>
        Const(v)

      case Apply(Select(lhs, op), List(rhs)) if op == "+" =>
        Add(toExprTerm(lhs), toExprTerm(rhs))

      case Apply(Select(lhs, op), List(rhs)) if op == "-" =>
        Sub(toExprTerm(lhs), toExprTerm(rhs))

      case Apply(Select(lhs, op), List(rhs)) if op == "*" =>
        Mul(toExprTerm(lhs), toExprTerm(rhs))

      case Apply(Select(lhs, op), List(rhs)) if op == "/" =>
        Div(toExprTerm(lhs), toExprTerm(rhs))
      
      case Apply(Select(_, "exp"), List(arg)) => Exp(toExprTerm(arg))
      
      case Select(arg, "unary_-") => Sub(Const(0.0), toExprTerm(arg))
      
      case id @ Ident(name) =>
        val actualTerm = proxyToOriginal.get(name) match
          case Some(original) => original  // Use the original variable
          case None => id  // Not a proxy, use as-is
        capturedVars(name) = actualTerm.asExprOf[Double]
        ConstSymbol(name)

      case other =>
        report.errorAndAbort(s"Unsupported term shape:\n${other.show}")

  def substituteVar(e: Expr): scala.quoted.Expr[Double => Double] =
    e match
      case Var(_) =>
        '{ (x: Double) => x }
      case Const(v) =>
        val vExpr = scala.quoted.Expr(v)
        '{ (x: Double) => $vExpr }
      case ConstSymbol(name) =>
        val captured = capturedVars(name)
        '{ (x: Double) => $captured }
      case Add(a, b) =>
        val fa = substituteVar(a)
        val fb = substituteVar(b)
        '{ (x: Double) => $fa(x) + $fb(x) }
      case Sub(a, b) =>
        val fa = substituteVar(a)
        val fb = substituteVar(b)
        '{ (x: Double) => $fa(x) - $fb(x) }
      case Mul(a, b) =>
        val fa = substituteVar(a)
        val fb = substituteVar(b)
        '{ (x: Double) => $fa(x) * $fb(x) }
      case Div(a, b) =>
        val fa = substituteVar(a)
        val fb = substituteVar(b)
        '{ (x: Double) => $fa(x) / $fb(x) }
      case Exp(a) =>
        val fa = substituteVar(a)
        '{ (x: Double) => scala.math.exp($fa(x)) }

  val symbolic: Expr = toExprTerm(bodyTerm)
  val derivative: Expr = simplify(diff(symbolic, "x"))
  
  substituteVar(derivative)

