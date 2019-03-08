package ffnn

import (
	"gonum.org/v1/gonum/mat"
	"../utils/matrices"
	"math"
	"io"
)

type FFLayer struct {
	// Size of the input this layer requires
	inputSize int
	// Size of the output this layer brings
	outputSize int
	// Current weights (and biases) as a matrix of:
	//   - `outputSize` rows
	//   - `inputSize`+1 columns
	weights *mat.Dense
	// Current inputs as a matrix of:
	//   - `inputSize`+1 rows (the last element will be 1)
	//   - just one column
	inputs *mat.Dense
	// Last calculated weighted inputs as z = wi + b, being a matrix of:
	//   - `outputSize` rows
	//   - just one column
	weightedInputs *mat.Dense
	// The activator function (will also hold its derivative
	activator Activator
	// This function is the activator for this layer. It will
	// Activations after running a = f(z), being a matrix of:
	//   - the same size of `weightedInputs`
	activations *mat.Dense
}

func makeFFLayer(inputSize, outputSize int, activator Activator, weights *mat.Dense) *FFLayer {
	// Creating an undefined input layer, aside from forcing it to 1 in the last element (the bias)
	inputs := mat.NewDense(inputSize + 1, 1, nil)
	inputs.Set(inputSize, 0, 1)
	// Creating an undefined weighted inputs layer
	weightedInputs := mat.NewDense(outputSize, 1, nil)
	// Creating an undefined activations layer
	activations := mat.NewDense(outputSize, 1, nil)

	return &FFLayer{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weights:        weights,
		inputs:         inputs,
		weightedInputs: weightedInputs,
		activator:      activator,
		activations:    activations,
	}
}

func newFFLayer(inputSize, outputSize int, activator Activator) *FFLayer {
	// Creating a noisy weights layer
	weights := matrices.Noise(outputSize, inputSize + 1, 1.0/math.Sqrt(float64(inputSize)))
	return makeFFLayer(inputSize, outputSize, activator, weights)
}

func loadFFLayer(inputSize, outputSize int, activator Activator, reader io.Reader) (*FFLayer, error) {
	// Loading the weights from a file
	weights := mat.NewDense(outputSize, inputSize + 1, nil)
	if _, err := weights.UnmarshalBinaryFrom(reader); err != nil {
		return nil, err
	} else {
		return makeFFLayer(inputSize, outputSize, activator, weights), nil
	}
}

func saveFFLayer(layer *FFLayer, writer io.Writer) error {
	_, err := layer.weights.MarshalBinaryTo(writer)
	return err
}

func decodeFFLayer(inputSize, outputSize int, activator Activator, data []byte) (*FFLayer, error) {
	// Loading the weights from memory
	weights := mat.NewDense(outputSize, inputSize + 1, nil)
	if err := weights.UnmarshalBinary(data); err != nil {
		return nil, err
	} else {
		return makeFFLayer(inputSize, outputSize, activator, weights), nil
	}
}

func encodeFFLayer(layer *FFLayer) ([]byte, error) {
	return layer.weights.MarshalBinary()
}

func (layer *FFLayer) InputSize() int {
	return layer.inputSize
}

func (layer *FFLayer) OutputSize() int {
	return layer.outputSize
}

func (layer *FFLayer) Weights() *mat.Dense {
	return layer.weights
}

func (layer *FFLayer) Inputs() *mat.Dense {
	return layer.inputs;
}

func (layer *FFLayer) WeightedInputs() *mat.Dense {
	return layer.weightedInputs
}

func (layer *FFLayer) Activator() Activator {
	return layer.activator
}

func (layer *FFLayer) Activations() *mat.Dense {
	return layer.activations
}

func (layer *FFLayer) Forward(inputs *mat.Dense) {
	// `inputs` will be a column, compatible with (inputSize, 1).
	// Fill the new inputs.
	for index := 0; index < layer.inputSize; index++ {
		layer.inputs.Set(index, 0, inputs.At(index, 0))
	}
	// Compute the weighted inputs from the current weights and inputs.
	layer.weightedInputs.Product(layer.weights, layer.inputs)
	// Compute the activations
	layer.activator.Base(layer.weightedInputs, layer.activations)
}
