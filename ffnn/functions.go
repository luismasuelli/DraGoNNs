package ffnn

import (
	"gonum.org/v1/gonum/mat"
	"../utils/matrices"
	"math"
)

// A simple mathematical function and its derivative
// It MUST operate element-wise, and so the matrices
//   WILL have matching dimensions
type Activator interface {
	// Function name (key)
	Name() string
	// The base function
	Base(*mat.Dense, *mat.Dense)
	// The derivative
	Derivative(source *mat.Dense, destination *mat.Dense)
}


// An example activator will be the Sigmoid
func sigmoid(i, j int, x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
type Sigmoid struct{}
func (s Sigmoid) Name() string {
	return "Sigmoid"
}
func (s Sigmoid) Base(source, destination *mat.Dense) {
	destination.Apply(sigmoid, source)
}
func (s Sigmoid) Derivative(source, destination *mat.Dense) {
	rows, columns := destination.Dims()
	ones := matrices.Fill(rows, columns, 1)
	base := mat.NewDense(rows, columns, nil)
	s.Base(source, base)
	destination.Sub(ones, base)
	destination.MulElem(destination, base)
}


// A simple mathematical function and its derivative.
// While the derivative takes expected and real output
//   and returns a matrix of values (one for each value),
//   the function by itself returns a single value being
//   the cost value (the actual error).
//
// This error metric is intended to be calculated on
//   the activations (final values) against the expected ones.
type ErrorMetric interface {
	// Function name (key)
	Name() string
	// The base function
	Base(finalActivations, expectedActivations *mat.Dense) float64
	// The derivative, intended to be computed into an existing matrix
	Gradient(finalActivations, expectedActivations, gradient *mat.Dense)
}


// An example error metric is the halved squared error
type HalfSquaredError struct{}
func (hse HalfSquaredError) Name() string {
	return "HalfSquaredError"
}
func (hse HalfSquaredError) Base(finalActivations, expectedActivations *mat.Dense) float64 {
	// This function is for a single training example. For batches of N elements,
	//   this value must be summed among them and then divided by N. This is the
	//   main reason why I avoided naming this one "mean squared error".
	rows, columns := expectedActivations.Dims()
	// This function is `1/2 * SUM((a - y)^2)`, being its derivative wrt `a` = `(a - y)`.
	// So just for the sake of being clean, this one is = 1/2 * SUM(gradients * gradients).
	// First, calculate the gradient and square it.
	difference := mat.NewDense(rows, columns, nil)
	hse.Gradient(finalActivations, expectedActivations, difference)
	difference.MulElem(difference, difference)
	// Now we sum the elements and halve them
	return mat.Sum(difference) / 2.0
}
func (hse HalfSquaredError) Gradient(finalActivations, expectedActivations, gradient *mat.Dense) {
	// The gradient of the HSE is the difference.
	gradient.Sub(expectedActivations, finalActivations)
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