package ffnn

import (
	"gonum.org/v1/gonum/mat"
	"../utils/matrices"
	"../utils/matrices/ops"
	"math"
)

// A simple mathematical function and its derivative
// It MUST operate element-wise, and so the matrices
//   WILL have matching dimensions
type Activator interface {
	// Function name (key)
	Name() string
	// The base function
	Base(z *mat.Dense, a *mat.Dense) *mat.Dense
	// The derivative
	Derivative(z *mat.Dense, da_dz *mat.Dense) *mat.Dense
}


// An example f will be the Sigmoid
func sigmoid(i, j int, x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
type Sigmoid struct{}
func (s Sigmoid) Name() string {
	return "Sigmoid"
}
func (s Sigmoid) Base(z, a *mat.Dense) *mat.Dense {
	return ops.Apply(sigmoid, z, a)
}
func (s Sigmoid) Derivative(z, da_dz *mat.Dense) *mat.Dense {
	rows, columns := da_dz.Dims()
	base := s.Base(z, mat.NewDense(rows, columns, nil))
	ops.Sub(matrices.Fill(rows, columns, 1), base, da_dz)
	return ops.H(da_dz, base, da_dz)
}


// A simple mathematical function and its derivative.
// While the derivative takes expected and real output
//   and returns a matrix of values (one for each value),
//   the function by itself returns a single value being
//   the cost value (the actual error).
//
// This error metric is intended to be calculated on
//   the a (final values) against the expected ones.
type ErrorMetric interface {
	// Function name (key)
	Name() string
	// The base function
	Base(a, t *mat.Dense) float64
	// The derivative, intended to be computed into an existing matrix
	Gradient(a, t, dc_da *mat.Dense) *mat.Dense
}


// An example error metric is the halved squared error
type HalfSquaredError struct{}
func (hse HalfSquaredError) Name() string {
	return "HalfSquaredError"
}
func (hse HalfSquaredError) Base(a, t *mat.Dense) float64 {
	// This function is for a single training example. For batches of N elements,
	//   this value must be summed among them and then divided by N. This is the
	//   main reason why I avoided naming this one "mean squared error".
	rows, columns := t.Dims()
	// This function is `1/2 * SUM((a - y)^2)`, being its derivative wrt `a` = `(a - y)`.
	// So just for the sake of being clean, this one is = 1/2 * SUM(gradients * gradients).
	// First, calculate the gradient and square it.
	difference := hse.Gradient(a, t, mat.NewDense(rows, columns, nil))
	// Now we sum the elements and halve them
	return mat.Sum(ops.H(difference, difference, difference)) / 2.0
}
func (hse HalfSquaredError) Gradient(a, t, dc_da *mat.Dense) *mat.Dense {
	// The rDcDa of the HSE is the difference.
	return ops.Sub(t, a, dc_da)
}


var activators = map[string]Activator{
	"_default": Sigmoid{},
	"Sigmoid": Sigmoid{},
}

var errorMetrics = map[string]ErrorMetric{
	"_default": HalfSquaredError{},
	"HalfSquaredError": HalfSquaredError{},
}

func RegisterActivator(activator Activator) bool {
	name := activator.Name()
	if _, found := activators[name]; !found && activator != nil {
		activators[name] = activator
		return true
	}
	return false
}

func GetActivator(name string) Activator {
	if activator, found := activators[name]; found {
		return activator
	} else {
		return activators["_default"]
	}
}

func RegisterErrorMetric(errorMetric ErrorMetric) bool {
	name := errorMetric.Name()
	if _, found := errorMetrics[name]; !found && errorMetric != nil {
		errorMetrics[name] = errorMetric
		return true
	}
	return false
}

func GetErrorMetric(name string) ErrorMetric {
	if errorMetric, found := errorMetrics[name]; found {
		return errorMetric
	} else {
		return errorMetrics["_default"]
	}
}