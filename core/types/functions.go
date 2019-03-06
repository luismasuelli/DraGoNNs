package types

import (
	"gonum.org/v1/gonum/mat"
	"../../utils/matrices"
	"math"
)

// A simple mathematical function and its derivative
// It MUST operate element-wise, and so the matrices
//   WILL have matching dimensions
type Activator interface {
	// The base function
	Base(*mat.Dense, *mat.Dense)
	// The derivative
	Derivative(*mat.Dense, *mat.Dense)
}


// An example activator will be the Sigmoid
func sigmoid(i, j int, x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
type Sigmoid struct{}
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
