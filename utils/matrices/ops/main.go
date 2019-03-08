package ops

import "gonum.org/v1/gonum/mat"

func H(a, b, result *mat.Dense) *mat.Dense {
	result.MulElem(a, b)
	return result
}

func Mul(a, b, result *mat.Dense) *mat.Dense {
	result.Product(a, b)
	return result
}

func Add(a, b, result *mat.Dense) *mat.Dense {
	result.Add(a, b)
	return result
}

func Sub(a, b, result *mat.Dense) *mat.Dense {
	result.Sub(a, b)
	return result
}

func Apply(f func(i, j int, v float64) float64, a, result *mat.Dense) *mat.Dense {
	result.Apply(f, a)
	return result
}