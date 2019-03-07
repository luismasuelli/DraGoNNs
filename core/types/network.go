package types

import "gonum.org/v1/gonum/mat"

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
	// While the cost gradients go for the output layer, these ones go
	//   for the middle layer(s).
	activationsCostGradients []*mat.Dense
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
func (network *SimpleFFNetwork) costGradientOverOutputActivations(layer *SimpleFFLayer, layerIndex int, expectedOutputActivations *mat.Dense) *mat.Dense {
	activationCostGradients := network.activationsCostGradients[layerIndex]
	// op1 Matrix size: (layer.outputSize rows, 1 column)
	// op2 Matrix size: (layer.outputSize rows, 1 column)
	// Result Matrix size: (layer.outputSize rows, 1 column)
	network.errorMetric.Gradient(layer.activations, expectedOutputActivations, activationCostGradients)
	// Return Matrix size: (layer.outputSize rows, 1 column)
	return activationCostGradients
}

// Recursive error calculation
func (network *SimpleFFNetwork) propagatedCostGradient(layerIndex int, nextLayerErrors *mat.Dense) *mat.Dense {
	nextLayer := network.layers[layerIndex + 1]
	nextLayerTransposedWeights := nextLayer.weights.T()
	activationCostGradients := network.activationsCostGradients[layerIndex]
	// Op1 Matrix Size: (nextLayer.inputSize = layer.outputSize rows, nextLayer.outputSize columns)
	// Op2 Matrix Size: (nextLayer.outputSize rows, 1 column)
	// Result Matrix Size: (nextLayer.inputSize = layer.outputSize rows, 1 column)
	activationCostGradients.Product(nextLayerTransposedWeights, network.errorsOverWeightedInputs[layerIndex + 1])
	// Return Matrix Size: (nextLayer.inputSize = layer.outputSize rows, 1 column)
	return activationCostGradients
}

// Derivative(layer.Activation)(layer.weightedInputs) -> stored in corresponding activator's derivative result
func (network *SimpleFFNetwork) activationDerivativeOverWeightedInputs(layer *SimpleFFLayer, layerIndex int) *mat.Dense {
	output := network.activatorDerivativeResultsOverWeightedInputs[layerIndex]
	// Op1 Matrix Size: (layer.outputSize rows, 1 column)
	// Result Matrix Size: (layer.outputSize rows, 1 column)
	layer.activator.Derivative(layer.weightedInputs, output)
	// Return Matrix Size: (layer.outputSize rows, 1 column)
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
	// Fetched Matrix size: (layer.outputSize rows, 1 column)
	costGradient := network.costGradientOverOutputActivations(lastLayer, lastLayerIndex, expectedOutputActivations)
	// Then we calculate the sigmoid prime over the last weighted inputs (which will have the same dimensions of the
	//   activations, and so the result will)
	// Fetched Matrix Size: (layer.outputSize rows, 1 column)
	activatorDerivativeResultOverWeightedInputs := network.activationDerivativeOverWeightedInputs(lastLayer, lastLayerIndex)
	// And finally we element-wise multiply the gradient with the derivative
	errorsOverWeightedInputs := network.errorsOverWeightedInputs[lastLayerIndex]
	// Op1 Matrix Size: (layer.outputSize rows, 1 column)
	// Op2 Matrix Size: (layer.outputSize rows, 1 column)
	// Result Matrix Size: (layer.outputSize rows, 1 column)
	errorsOverWeightedInputs.MulElem(costGradient, activatorDerivativeResultOverWeightedInputs)
	// And return such matrix
	// Return Matrix Size: (layer.outputSize rows, 1 column)
	return errorsOverWeightedInputs
}

// This is the second, and more, differential error(s) being calculated. It will imply the weights of the following
//   layer, and the errors from the following layer.
func (network *SimpleFFNetwork) differentialErrorsFromFollowingLayer(
	layerIndex int,
) *mat.Dense {
	layer := network.layers[layerIndex]
	// First, we calculate the propagated gradient by using the next layer errors and transposing the next layer weights
	// Op2 Matrix Size: (nextLayer.outputSize rows, 1 column)
	// Fetched Matrix Size: (nextLayer.inputSize = layer.outputSize rows, 1 column)
	activationCostGradients := network.propagatedCostGradient(layerIndex, network.errorsOverWeightedInputs[layerIndex + 1])
	// Then, we have a matching matrix of propagated gradients. Just calculate the derivative
	// Fetched Matrix Size: (layer.outputSize rows, 1 column)
	activatorDerivativeResultOverWeightedInputs := network.activationDerivativeOverWeightedInputs(layer, layerIndex)
	// And finally we element-wise multiply the propagated gradient with the derivative
	// Op1 Matrix Size: (nextLayer.inputSize = layer.outputSize rows, 1 column)
	// Op2 Matrix Size: (layer.outputSize rows, 1 column)
	// Result Matrix Size: (layer.outputSize rows, 1 column)
	errorsOverWeightedInputs := network.errorsOverWeightedInputs[layerIndex]
	errorsOverWeightedInputs.MulElem(activationCostGradients, activatorDerivativeResultOverWeightedInputs)
	// And return such matrix
	// Return Matrix Size: (layer.outputSize rows, 1 column)
	return errorsOverWeightedInputs
}

// Now, to fix the layers!
func (network *SimpleFFNetwork) fixLayer(layerIndex int, learningRate float64) {
	layer := network.layers[layerIndex]
	weights := layer.weights

	// Cartesian product of inputs and errors
	inputs := layer.inputs.T()
	errors := network.errorsOverWeightedInputs[layerIndex]
	rows, _ := errors.Dims() // rows = n. of errors (neurons)
	_, columns := inputs.Dims() // columns = n. of inputs (or former activations)
	errorOnInputs := mat.NewDense(rows, columns, nil) // size = n. of errors * n. of inputs
	// Op1 Matrix Size: (layer.outputSize rows, 1 column)
	// Op2 Matrix Size: (1 row, layer.inputSize columns)
	// Result Matrix Size: (layer.outputSize rows, layer.inputSize column)
	errorOnInputs.Product(errors, inputs)
	// Scaling the errors by the learning rate
	errorOnInputs.Scale(learningRate, errorOnInputs)
	// Finally, modify the widths by subtracting the scaled errors
	weights.Sub(weights, errorOnInputs)
}

func (network *SimpleFFNetwork) TrainWithRate(input *mat.Dense, expectedOutput *mat.Dense, learningRate float64) (*mat.Dense, float64) {
	// Get the outputs by running a normal forward, and the cost (absolute error)
	network.Forward(input)
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

func (network *SimpleFFNetwork) Train(input *mat.Dense, expectedOutput *mat.Dense) (*mat.Dense, float64) {
	return network.TrainWithRate(input, expectedOutput, network.defaultLearningRate)
}