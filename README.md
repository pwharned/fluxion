# Fluxion

Automatic differentiation at compile time using Scala 3 macros. Derivatives are computed during compilation and emitted as optimized native code.

## How it works

Write your loss function inline:

```scala
inline def loss(w: Double, x: Double, y: Double): Double =
  val pred = 1.0 / (1.0 + math.exp(-(w * x)))
  (pred - y) * (pred - y)
```

Get the derivative for free:

```scala
val gradFn = diffFn((w: Double) => loss(w, x, y))
val gradient = gradFn(currentWeight)  // Just a function call
```

The macro extracts the mathematical structure, applies differentiation rules (chain rule, product rule, etc.), simplifies the expression, and generates native code. No runtime overhead.

## Example: Logistic Regression

Binary classification on Iris dataset (99 samples, 2 features):

```scala
// Define loss w.r.t. each parameter
inline def lossW0(w0: Double, w1: Double, bias: Double, x0: Double, x1: Double, y: Double): Double =
  val z = w0 * x0 + w1 * x1 + bias
  val pred = 1.0 / (1.0 + math.exp(-z))
  (pred - y) * (pred - y)

// Training loop
for epoch <- 0 until 100 do
  for (x, y) <- data do
    val grad = diffFn((w: Double) => lossW0(w, w1, bias, x(0), x(1), y))
    w0 -= learningRate * grad(w0)
    // ... same for w1, bias
```

Converges to 98.99% accuracy in 100 epochs.

## Benchmarks

Compared against PyTorch on the same task (M1 Mac, Scala Native 0.5.0):

| Metric | Fluxion | PyTorch | Speedup |
|--------|---------|---------|---------|
| Training (100 epochs) | 2.67 ms | 133.84 ms | 50x |
| Inference (single prediction) | 0.07 μs | 7.63 μs | 109x |
| Inference (10k predictions) | 0.65 ms | 76.25 ms | 117x |

Results are identical (loss: 0.1444, accuracy: ~99%).

The difference comes from:
- Native code vs Python/JIT overhead
- Direct function calls vs tensor abstractions
- Zero autograd tracking at runtime
- Compile-time vs runtime differentiation

## Supported Operations

- Arithmetic: `+`, `-`, `*`, `/`
- Exponential: `math.exp`
- Unary minus: `-x`

Adding more operations is straightforward - just pattern match in the macro and add the differentiation rule.

## Limitations

- Loss functions must be `inline def` so the macro can see their body
- No GPU support (CPU native code only)
- Manual parameter-by-parameter differentiation (no automatic gradient accumulation like PyTorch)
- Simple expression tree only (no control flow, recursion, etc.)

## Building

Requires Scala 3.3+ and Scala Native:

```bash
sbt run
```

Compilation includes LLVM optimization, which is where most time is spent (~6 seconds).

## Use Cases

Best for:
- Edge deployment where binary size and startup time matter
- Microservices needing minimal latency
- Simple gradient descent on custom loss functions
- Learning how AD works under the hood
