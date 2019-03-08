package ffnn

import (
	"gonum.org/v1/gonum/mat"
	"../utils/matrices"
	"../utils/matrices/ops"
	"math"
	"errors"
	"fmt"
)

type FFLayer struct {
	// Layer Spec

	// Size of the input this layer requires
	inputSize int
	// Size of the output this layer brings
	outputSize int
	// The f function (will also hold its derivative
	f Activator

	// Layer State

	// Meaning: current weights
	// Size: outputSize x inputSize
	w *mat.Dense
	// Meaning: current biases
	// Size: outputSize x 1
	b *mat.Dense
	// Meaning: current inputs
	// Size: inputSize x 1
	i *mat.Dense
	// Meaning: linear composition wi
	// Size: outputSize x 1
	wi *mat.Dense
	// Meaning: linear composition wi + b
	// size: outputSize x 1
	z *mat.Dense
	// Meaning: activations f(z)
	// Size: outputSize x 1
	a *mat.Dense
}

func makeFFLayer(inputSize, outputSize int, activator Activator, w *mat.Dense, b *mat.Dense) *FFLayer {
	// Creating an undefined input layer, aside from forcing it to 1 in the last element (the bias)
	i := mat.NewDense(inputSize, 1, nil)
	// Creating an undefined weighted i layer
	wi := mat.NewDense(outputSize, 1, nil)
	// Creating an undefined weighted i layer
	z := mat.NewDense(outputSize, 1, nil)
	// Creating an undefined a layer
	a := mat.NewDense(outputSize, 1, nil)

	return &FFLayer{
		inputSize:  inputSize,
		outputSize: outputSize,
		f:          activator,
		w:          w,
		b:          b,
		i:          i,
		wi:         wi,
		z:          z,
		a:          a,
	}
}

func newFFLayer(inputSize, outputSize int, activator Activator) *FFLayer {
	bound := 1.0/math.Sqrt(float64(inputSize))
	// Creating a noisy w/b layer
	w := matrices.Noise(outputSize, inputSize, bound)
	b := matrices.Noise(outputSize, 1, bound)
	return makeFFLayer(inputSize, outputSize, activator, w, b)
}

func unMarshall(expectedRows, expectedColumns int, data []byte, element string) (*mat.Dense, error) {
	m := mat.NewDense(expectedRows, expectedColumns, nil)
	m.Reset()
	if err := m.UnmarshalBinary(data); err != nil {
		return nil, err
	} else {
		rows, columns := m.Dims()
		if rows != expectedRows || columns != expectedColumns {
			return nil, errors.New(fmt.Sprintf(
				"layer %v size mismatch between requested and unmarshaled", element,
			))
		}
	}
	return m, nil
}

func decodeFFLayer(inputSize, outputSize int, activator Activator, wData, bData []byte) (*FFLayer, error) {
	// Loading the w from memory
	var w, b *mat.Dense
	var err error
	if w, err = unMarshall(outputSize, inputSize, wData, "weights"); err != nil {
		return nil, err
	}
	if b, err = unMarshall(outputSize, 1, bData, "biases"); err != nil {
		return nil, err
	}
	return makeFFLayer(inputSize, outputSize, activator, w, b), nil
}

func encodeFFLayer(layer *FFLayer) ([]byte, []byte, error, error) {
	w, errW := layer.w.MarshalBinary()
	b, errB := layer.b.MarshalBinary()
	return w, b, errW, errB
}

func (layer *FFLayer) InputSize() int {
	return layer.inputSize
}

func (layer *FFLayer) OutputSize() int {
	return layer.outputSize
}

func (layer *FFLayer) Weights() *mat.Dense {
	return layer.w
}

func (layer *FFLayer) Inputs() *mat.Dense {
	return layer.i;
}

func (layer *FFLayer) WeightedInputs() *mat.Dense {
	return layer.z
}

func (layer *FFLayer) Activator() Activator {
	return layer.f
}

func (layer *FFLayer) Activations() *mat.Dense {
	return layer.a
}

func (layer *FFLayer) Forward(inputs *mat.Dense) {
	// `i` will be a column, compatible with (inputSize, 1).
	// Fill the new i.
	layer.i.Copy(inputs)
	// Compute the a = f(wi + b).
	layer.f.Base(ops.Add(ops.Mul(layer.w, layer.i, layer.wi), layer.b, layer.z), layer.a)
}
