package matrices

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
)

func Fill(rows, columns int, value float64) *mat.Dense {
	elements := make([]float64, rows * columns)
	for index := range elements {
		elements[index] = value
	}
	return mat.NewDense(rows, columns, elements)
}

func FillRow(columns int, value float64) *mat.Dense {
	return Fill(1, columns, value)
}

func FillColumn(rows int, value float64) *mat.Dense {
	return Fill(rows, 1, value)
}

func Noise(rows, columns int, cap float64) *mat.Dense {
	elements := make([]float64, rows * columns)
	cap = math.Abs(cap)
	random := distuv.Uniform{Min: -cap, Max: cap}
	for index := range elements {
		elements[index] = random.Rand()
	}
	return mat.NewDense(rows, columns, elements)
}

func NoiseRow(columns int, cap float64) *mat.Dense {
	return Noise(1, columns, cap)
}

func NoiseColumn(rows int, cap float64) *mat.Dense {
	return Noise(rows, 1, cap)
}
