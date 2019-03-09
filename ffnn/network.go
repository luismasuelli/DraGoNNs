package ffnn

import (
	"gonum.org/v1/gonum/mat"
	"../utils/matrices/ops"
)

type FFNetwork struct {
	// The layers, in strict order.
	layers []*FFLayer
	// ********************************
	// Training-related fields start here.
	// ********************************
	// The default learning rate, needed for training.
	defaultLearningRate float64
	// The cost (error) function for the training.
	c ErrorMetric
	// These hold the gradient costs for the layers; these ones go
	//   for all the layer(s).
	rDcDa []*mat.Dense
	// Per-layer f-derivative over weighted i. Will have
	//  the sizes of corresponding layers' weighted i.
	rDaDz []*mat.Dense
	// Will have the sizes of corresponding layers' weighted i.
	// It is = dc/dz = dc/da (*) da/dz.
	delta []*mat.Dense
}

func (network *FFNetwork) Layer(index int) *FFLayer {
	return network.layers[index]
}

func (network *FFNetwork) DefaultLearningRate() float64 {
	return network.defaultLearningRate
}

func (network *FFNetwork) Forward(input *mat.Dense) *mat.Dense {
	for _, layer := range network.layers {
		layer.Forward(input)
		input = layer.a
	}
	// After this, all the data will be available inside each layer
	// And the paradoxical part is that `i` will hold the outputs
	//   in the end
	return input
}

// Gradient(network.c)(layer.a, expected) -> stored in networks' output a cost gradient
func (network *FFNetwork) opDcDaInLastLayer(layer *FFLayer, layerIndex int, t *mat.Dense) *mat.Dense {
	// op1 Matrix size: (layer.outputSize rows, 1 column)
	// op2 Matrix size: (layer.outputSize rows, 1 column)
	// Result Matrix size: (layer.outputSize rows, 1 column)
	return network.c.Gradient(layer.a, t, network.rDcDa[layerIndex])
}

// Recursive error calculation
func (network *FFNetwork) opDcDaInNonLastLayer(layerIndex int, nextLayerErrors *mat.Dense) *mat.Dense {
	// Op1 Matrix Size: (nextLayer.inputSize = layer.outputSize rows, nextLayer.outputSize columns)
	// Op2 Matrix Size: (nextLayer.outputSize rows, 1 column)
	// Result Matrix Size: (nextLayer.inputSize = layer.outputSize rows, 1 column)
	return ops.Mul(network.layers[layerIndex + 1].w.T(), network.delta[layerIndex + 1], network.rDcDa[layerIndex])
}

// Derivative(layer.Activation)(layer.z) -> stored in corresponding f's derivative result
func (network *FFNetwork) opDaDz(layer *FFLayer, layerIndex int) *mat.Dense {
	// Op1 Matrix Size: (layer.outputSize rows, 1 column)
	// Result Matrix Size: (layer.outputSize rows, 1 column)
	return layer.f.Derivative(layer.z, network.rDaDz[layerIndex])
}

// This is the first differential error being calculated. It will imply the gradient function over the costs.
func (network *FFNetwork) opDeltaInLastLayer(
	lastLayerIndex int, expectedOutputActivations *mat.Dense,
) *mat.Dense {
	// Consider z = weighted i
	//          a = final output a
	//          C = the cost function
	//          differential error on the output = gradient of C with respect to a
	//          differential error on the weighted i = element-wise differential error on the output * (
	//            derivative of Activation function over the weighted input for that output
	//          )
	lastLayer := network.layers[lastLayerIndex]
	// First, we calculate the gradient of C by the a using our particular final output a
	// Fetched Matrix size: (layer.outputSize rows, 1 column)
	rDcDa := network.opDcDaInLastLayer(lastLayer, lastLayerIndex, expectedOutputActivations)
	// Then we calculate the sigmoid prime over the last weighted i (which will have the same dimensions of the
	//   a, and so the result will)
	// Fetched Matrix Size: (layer.outputSize rows, 1 column)
	rDaDz := network.opDaDz(lastLayer, lastLayerIndex)
	// And finally we element-wise multiply the gradient with the derivative
	// Op1 Matrix Size: (layer.outputSize rows, 1 column)
	// Op2 Matrix Size: (layer.outputSize rows, 1 column)
	// Result Matrix Size: (layer.outputSize rows, 1 column)
	return ops.H(rDcDa, rDaDz, network.delta[lastLayerIndex])
}

// This is the second, and more, differential error(s) being calculated. It will imply the w of the following
//   layer, and the errors from the following layer.
func (network *FFNetwork) opDeltaInNonLastLayer(
	layerIndex int,
) *mat.Dense {
	layer := network.layers[layerIndex]
	// First, we calculate the propagated gradient by using the next layer errors and transposing the next layer w
	// Op2 Matrix Size: (nextLayer.outputSize rows, 1 column)
	// Fetched Matrix Size: (nextLayer.inputSize = layer.outputSize rows, 1 column)
	rDcDa := network.opDcDaInNonLastLayer(layerIndex, network.delta[layerIndex + 1])
	// Then, we have a matching matrix of propagated gradients. Just calculate the derivative
	// Fetched Matrix Size: (layer.outputSize rows, 1 column)
	rDaDz := network.opDaDz(layer, layerIndex)
	// And finally we element-wise multiply the propagated gradient with the derivative
	// Op1 Matrix Size: (nextLayer.inputSize = layer.outputSize rows, 1 column)
	// Op2 Matrix Size: (layer.outputSize rows, 1 column)
	// Result Matrix Size: (layer.outputSize rows, 1 column)
	return ops.H(rDcDa, rDaDz, network.delta[layerIndex])
}

// Now, to fix the layers!
func (network *FFNetwork) fixLayer(layerIndex int, learningRate float64) {
	layer := network.layers[layerIndex]
	weights := layer.w
	biases := layer.b

	// Cartesian product of i and delta
	iT := layer.i.T()
	delta := network.delta[layerIndex]
	rows, _ := delta.Dims() // rows = n. of errors (neurons)
	_, columns := iT.Dims() // columns = n. of inputs (or former a)
	deltaXiT := mat.NewDense(rows, columns, nil) // size = n. of errors x n. of inputs
	delta_ := mat.NewDense(rows, 1, nil) // size = n. of errors x n. of inputs
	// Op1 Matrix Size: (layer.outputSize rows, 1 column)
	// Op2 Matrix Size: (1 row, layer.inputSize columns)
	// Result Matrix Size: (layer.outputSize rows, layer.inputSize column)
	// Finally, modify the widths and bias by subtracting the scaled delta
	weights.Sub(weights, ops.Scale(learningRate, ops.Mul(delta, iT, deltaXiT), deltaXiT))
	biases.Sub(biases, ops.Scale(learningRate, delta, delta_))
}

func (network *FFNetwork) Test(input *mat.Dense, expectedOutput *mat.Dense) (*mat.Dense, float64) {
	// Get the outputs by running a normal forward, and the cost (absolute error)
	output := network.Forward(input)
	return output, network.c.Base(output, expectedOutput)
}

func (network *FFNetwork) adjust(expectedOutput *mat.Dense, learningRate float64) {
	layersCount := len(network.layers)
	network.opDeltaInLastLayer(layersCount - 1, expectedOutput)
	for index := layersCount - 2; index >= 0; index-- {
		network.opDeltaInNonLastLayer(index)
	}
	// And finally, after we know all the errors (which are vertical rows), fix the layers
	for index := 0; index < layersCount; index++ {
		network.fixLayer(index, learningRate)
	}
}

func (network *FFNetwork) TrainWithRate(input *mat.Dense, expectedOutput *mat.Dense, learningRate float64) (*mat.Dense, float64) {
	// Get the outputs by running a normal forward, and the cost (absolute error)
	output, cost := network.Test(input, expectedOutput)
	// Now compute the errors backward, and adjust using a learning rate
	network.adjust(expectedOutput, learningRate)
	return output, cost
}

func (network *FFNetwork) Train(input *mat.Dense, expectedOutput *mat.Dense) (*mat.Dense, float64) {
	return network.TrainWithRate(input, expectedOutput, network.defaultLearningRate)
}
