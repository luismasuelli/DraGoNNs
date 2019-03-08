package ops

import "gonum.org/v1/gonum/mat"

func H(a, b mat.Matrix, result *mat.Dense) *mat.Dense {
	result.MulElem(a, b)
	return result
}

func Mul(a, b mat.Matrix, result *mat.Dense) *mat.Dense {
	result.Product(a, b)
	return result
}

func Add(a, b mat.Matrix, result *mat.Dense) *mat.Dense {
	result.Add(a, b)
	return result
}

func Sub(a, b mat.Matrix, result *mat.Dense) *mat.Dense {
	result.Sub(a, b)
	return result
}

func Scale(f float64, a mat.Matrix, result *mat.Dense) *mat.Dense {
	result.Scale(f, a)
	return result
}

func Apply(f func(i, j int, v float64) float64, a mat.Matrix, result *mat.Dense) *mat.Dense {
	result.Apply(f, a)
	return result
}