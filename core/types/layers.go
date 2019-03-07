package types

import (
	"gonum.org/v1/gonum/mat"
	"../../utils/matrices"
	"math"
	"io"
)

type SimpleFFLayer struct {
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

func newSimpleFFLayer(inputSize, outputSize int, activator Activator, weights *mat.Dense) *SimpleFFLayer {
	// Creating an undefined input layer, aside from forcing it to 1 in the last element (the bias)
	inputs := mat.NewDense(inputSize + 1, 1, nil)
	inputs.Set(inputSize, 0, 1)
	// Creating an undefined weighted inputs layer
	weightedInputs := mat.NewDense(outputSize, 1, nil)
	// Creating an undefined activations layer
	activations := mat.NewDense(outputSize, 1, nil)

	return &SimpleFFLayer{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weights:        weights,
		inputs:         inputs,
		weightedInputs: weightedInputs,
		activator:      activator,
		activations:    activations,
	}
}

func newSimpleFFLayerFromNoise(inputSize, outputSize int, activator Activator) *SimpleFFLayer {
	// Creating a noisy weights layer
	weights := matrices.Noise(outputSize, inputSize + 1, 1.0/math.Sqrt(float64(inputSize)))
	return newSimpleFFLayer(inputSize, outputSize, activator, weights)
}

func newSimpleFFLayerFromStream(inputSize, outputSize int, activator Activator, reader io.Reader) *SimpleFFLayer {
	// Loading the weights from file
	weights := mat.NewDense(outputSize, 1, nil)
	weights.UnmarshalBinaryFrom(reader)
	return newSimpleFFLayer(inputSize, outputSize, activator, weights)
}

func newSimpleFFLayerFromMemory(inputSize, outputSize int, activator Activator, data []byte) *SimpleFFLayer {
	// Loading the weights from memory
	weights := mat.NewDense(outputSize, 1, nil)
	weights.UnmarshalBinary(data)
	return newSimpleFFLayer(inputSize, outputSize, activator, weights)
}

func (layer *SimpleFFLayer) InputSize() int {
	return layer.inputSize
}

func (layer *SimpleFFLayer) OutputSize() int {
	return layer.outputSize
}

func (layer *SimpleFFLayer) Weights() *mat.Dense {
	return layer.weights
}

func (layer *SimpleFFLayer) Inputs() *mat.Dense {
	return layer.inputs;
}

func (layer *SimpleFFLayer) WeightedInputs() *mat.Dense {
	return layer.weightedInputs
}

func (layer *SimpleFFLayer) Activator() Activator {
	return layer.activator
}

func (layer *SimpleFFLayer) Activations() *mat.Dense {
	return layer.activations
}

func (layer *SimpleFFLayer) Forward(inputs *mat.Dense) {
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

type SimpleFFNetwork struct {
	// The layers, in strict order.
	layers []*SimpleFFLayer
	// ********************************
	// Training-related fields start here.
	// ********************************
	// The default learning rate, needed for training.
	defaultLearningRate float64
	// The cost (error) function for the training.
	errorMetric ErrorMetric
	// The cost gradients at output activations. Will be a matrix of:
	//   - `outputSize` (of the last layer) rows
	//   - just one column
	outputActivationsCostGradient *mat.Dense
	// While the cost gradients go for the output layer, these ones go
	//   for the middle layer(s).
	middleActivationsCostGradients []*mat.Dense
	// Per-layer activator-derivative over weighted inputs. Will have
	//  the sizes of corresponding layers' weighted inputs.
	activatorDerivativeResultsOverWeightedInputs []*mat.Dense
	// Will have the sizes of corresponding layers' weighted inputs.
	errorsOverWeightedInputs []*mat.Dense
}

func (network *SimpleFFNetwork) Layer(index int) *SimpleFFLayer {
	return network.layers[index]
}

func (network *SimpleFFNetwork) DefaultLearningRate() float64 {
	return network.defaultLearningRate
}

func (network *SimpleFFNetwork) Forward(input *mat.Dense) {
	for _, layer := range network.layers {
		layer.Forward(input)
		input = layer.activations
	}
	// After this, all the data will be available inside each layer
}

// Gradient(network.errorMetric)(layer.activations, expected) -> stored in networks' output activations cost gradient
func (network *SimpleFFNetwork) costGradientOverOutputActivations(layer *SimpleFFLayer, expectedOutputActivations *mat.Dense) *mat.Dense {
	network.errorMetric.Gradient(layer.activations, expectedOutputActivations, network.outputActivationsCostGradient)
	return network.outputActivationsCostGradient
}

// Recursive error
func (network *SimpleFFNetwork) propagatedCostGradient(layerIndex int, nextLayerErrors *mat.Dense) *mat.Dense {
	nextLayer := network.layers[layerIndex + 1]
	nextLayerTransposedWeights := nextLayer.weights.T()
	activationCostGradients := network.middleActivationsCostGradients[layerIndex]
	activationCostGradients.Product(nextLayerTransposedWeights, network.errorsOverWeightedInputs[layerIndex + 1])
	return activationCostGradients
}

// Derivative(layer.Activation)(layer.weightedInputs) -> stored in corresponding activator's derivative result
func (network *SimpleFFNetwork) activationDerivativeOverWeightedInputs(layer *SimpleFFLayer, layerIndex int) *mat.Dense {
	output := network.activatorDerivativeResultsOverWeightedInputs[layerIndex]
	layer.activator.Derivative(layer.weightedInputs, output)
	return output
}

// This is the first differential error being calculated. It will imply the gradient function over the costs.
func (network *SimpleFFNetwork) differentialErrorFromOutputs(
	lastLayerIndex int, expectedOutputActivations *mat.Dense,
) *mat.Dense {
	// Consider z = weighted inputs
	//          a = final output activations
	//          C = the cost function
	//          differential error on the output = gradient of C with respect to a
	//          differential error on the weighted inputs = element-wise differential error on the output * (
	//            derivative of Activation function over the weighted input for that output
	//          )
	lastLayer := network.layers[lastLayerIndex]
	// First, we calculate the gradient of C by the activations using our particular final output activations
	costGradient := network.costGradientOverOutputActivations(lastLayer, expectedOutputActivations)
	// Then we calculate the sigmoid prime over the last weighted inputs (which will have the same dimensions of the
	//   activations, and so the result will)
	activatorDerivativeResultOverWeightedInputs := network.activationDerivativeOverWeightedInputs(lastLayer, lastLayerIndex)
	// And finally we element-wise multiply the gradient with the derivative
	errorsOverWeightedInputs := network.errorsOverWeightedInputs[lastLayerIndex]
	errorsOverWeightedInputs.MulElem(costGradient, activatorDerivativeResultOverWeightedInputs)
	// And return such matrix
	return errorsOverWeightedInputs
}

// This is the second, and more, differential error(s) being calculated. It will imply the weights of the following
//   layer, and the errors from the following layer.
func (network *SimpleFFNetwork) differentialErrorsFromFollowingLayer(
	layerIndex int,
) *mat.Dense {
	layer := network.layers[layerIndex]
	// First, we calculate the propagated gradient by using the next layer errors and transposing the next layer weights
	activationCostGradients := network.propagatedCostGradient(layerIndex, network.errorsOverWeightedInputs[layerIndex + 1])
	// Then, we have a matching matrix of propagated gradients. Just calculate the derivative
	activatorDerivativeResultOverWeightedInputs := network.activationDerivativeOverWeightedInputs(layer, layerIndex)
	// And finally we element-wise multiply the propagated gradient with the derivative
	errorsOverWeightedInputs := network.errorsOverWeightedInputs[layerIndex]
	errorsOverWeightedInputs.MulElem(activationCostGradients, activatorDerivativeResultOverWeightedInputs)
	// And return such matrix
	return errorsOverWeightedInputs
}

// Now, to fix the layers!
func (network *SimpleFFNetwork) fixLayer(layerIndex int, learningRate float64) {
	layer := network.layers[layerIndex]
	weights := layer.weights

	// Cartesian product of inputs and errors
	inputs := layer.inputs
	errors := network.errorsOverWeightedInputs[layerIndex]
	rows, _ := errors.Dims() // rows = n. of errors (neurons)
	columns, _ := inputs.Dims() // columns = n. of inputs (or former activations)
	errorOnInputs := mat.NewDense(rows, columns, nil) // size = n. of errors * n. of inputs
	errorOnInputs.Product(errors, inputs.T())
	// Scaling the errors by the learning rate
	errorOnInputs.Scale(learningRate, errorOnInputs)
	// Finally, modify the widths by subtracting the scaled errors
	weights.Sub(weights, errorOnInputs)
}

func (network *SimpleFFNetwork) TrainWithRate(input *mat.Dense, expectedOutput *mat.Dense, learningRate float64) (*mat.Dense, float64) {
	// Get the outputs by running a normal forward, and the cost (absolute error)
	network.Forward(input);
	layersCount := len(network.layers)
	output := network.layers[layersCount - 1].activations
	cost := network.errorMetric.Base(output, expectedOutput)
	// Now compute the errors backward, and adjust using a learning rate
	network.differentialErrorFromOutputs(layersCount - 1, expectedOutput)
	for index := layersCount - 2; index >= 0; index++ {
		network.differentialErrorsFromFollowingLayer(index)
	}
	// And finally, after we know all the errors (which are vertical rows), fix the layers
	for index := 0; index < layersCount; index++ {
		network.fixLayer(index, learningRate)
	}
	return output, cost
}