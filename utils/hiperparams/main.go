package hiperparams

type Shape struct {
	InputSize, OutputSize int
}

func ShapeChain(sizes ...int) []Shape {
	if (len(sizes) < 2) {
		panic("at least 2 values must be given to make the shapes")
	}
	limit := len(sizes) - 1
	if sizes[limit] < 1 {
		panic("sizes must be strictly positive")
	}
	shapes := make([]Shape, limit)
	for index := 0; index < limit; index++ {
		size := sizes[index]
		if size < 1 {
			panic("sizes must be strictly positive")
		}
		shapes[index] = Shape{size, sizes[index+1]}
	}
	return shapes
}