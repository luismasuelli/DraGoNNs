package matrices

import "gonum.org/v1/gonum/mat"

// Standard ops and element-wise operations

func Add(a, b mat.Matrix) mat.Matrix {
	rows, columns := a.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Add(a, b)
	return result
}

func Sub(a, b mat.Matrix) mat.Matrix {
	rows, columns := a.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Sub(a, b)
	return result
}

func Mul(a, b mat.Matrix) mat.Matrix {
	rows, columns := a.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.MulElem(a, b)
	return result
}

func Scale(a float64, b mat.Matrix) mat.Matrix {
	rows, columns := b.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Scale(a, b)
	return result
}

// Linear product, ideal for AX + B operations

func LinearProduct(a, b mat.Matrix) mat.Matrix {
	rows, _ := a.Dims()
	_, columns := b.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Product(a, b)
	return result
}

// Other operations to modify the matrices

func GlueHorizontally(a, b mat.Matrix) mat.Matrix {
	rows, columnsA := a.Dims()
	_, columnsB := b.Dims()
	result := mat.NewDense(rows, columnsA + columnsB, nil)
	result.Augment(a, b)
	return result
}

func GlueVertically(a, b mat.Matrix) mat.Matrix {
	rowsA, columns := a.Dims()
	rowsB, _ := b.Dims()
	result := mat.NewDense(rowsA + rowsB, columns, nil)
	result.Augment(a, b)
	return result
}