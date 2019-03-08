package ffnn

import (
	"os"
	"encoding/json"
	"errors"
	"strings"
	"gonum.org/v1/gonum/mat"
)


type serializedFFLayer struct {
	activator           string
	outputSize          int
	weightsData         []byte
}
type serializedFFNetwork struct {
	errorMetric         string
	defaultLearningRate float64
	inputSize           int
	layers              []serializedFFLayer
}
func withExtension(filename string, extension string) string {
	if strings.Trim(filename, " \r\n\t") == "" {
		return ""
	}

	if !strings.HasSuffix(filename, "." +extension) {
		filename += "." + extension
	}
	return filename
}
func loadLayer(inputSize int, outputSize int, activator Activator, marshaled []byte) (*FFLayer, error) {
	// Read everything
	return decodeFFLayer(inputSize, outputSize, activator, marshaled)
}


func Load(filename string) (*FFNetwork, error) {
	filename = withExtension(filename, "ffnn")
	if filename == "" {
		return nil, errors.New("filename is empty")
	}

	// Open file for reading
	var file *os.File
	var err error
	if file, err = os.Open(filename); err != nil {
		return nil, err
	} else {
		defer file.Close()
	}

	var serialized serializedFFNetwork
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&serialized); err != nil {
		return nil, err
	}

	if serialized.inputSize < 1 {
		return nil, errors.New("input size must be >= 1")
	}

	layersCount := len(serialized.layers)
	if len(serialized.layers) == 0 {
		return nil, errors.New("at least one layer must be present")
	}

	if serialized.defaultLearningRate <= 0 {
		return nil, errors.New("learning rate must be positive (and, preferably, small)")
	}

	network := &FFNetwork{
		defaultLearningRate: serialized.defaultLearningRate,
		errorMetric: GetErrorMetric(serialized.errorMetric),
		layers: make([]*FFLayer, layersCount),
		activatorDerivativeResultsOverWeightedInputs: make([]*mat.Dense, layersCount),
		activationsCostGradients: make([]*mat.Dense, layersCount),
		errorsOverWeightedInputs: make([]*mat.Dense, layersCount),
	}

	inputSize := serialized.inputSize
	for index, serializedLayer := range serialized.layers {
		outputSize := serializedLayer.outputSize
		if outputSize < 1 {
			return nil, errors.New("output size must be >= 1")
		}
		activator := GetActivator(serializedLayer.activator)

		if layer, err := loadLayer(inputSize, outputSize, activator, serializedLayer.weightsData); err != nil {
			return nil, err
		} else {
			network.layers[index] = layer
			// here we create the training matrices
			network.activatorDerivativeResultsOverWeightedInputs[index] = mat.NewDense(outputSize, 1, nil)
			network.activationsCostGradients[index] = mat.NewDense(outputSize, 1, nil)
			network.errorsOverWeightedInputs[index] = mat.NewDense(outputSize, 1, nil)
		}

		// output size is the new input size
		inputSize = serializedLayer.outputSize
	}

	return network, nil
}


func Save(network *FFNetwork, filename string) (error) {
	filename = withExtension(filename, ".ffnn")
	if network == nil {
		return errors.New("network is nil")
	}

	// Open file for writing
	var file *os.File
	var err error
	if file, err = os.Create(filename); err != nil {
		return err
	} else {
		defer file.Close()
	}

	serialized := serializedFFNetwork{
		defaultLearningRate: network.defaultLearningRate,
		inputSize: network.layers[0].inputSize,
		layers: make([]serializedFFLayer, len(network.layers)),
		errorMetric: network.errorMetric.Name(),
	}
	for index, layer := range network.layers {
		if weightsData, err := encodeFFLayer(layer); err != nil {
			return err
		} else {
			serialized.layers[index] = serializedFFLayer{
				activator: layer.activator.Name(),
				outputSize: layer.outputSize,
				weightsData: weightsData,
			}
		}
	}

	encoder := json.NewEncoder(file)
	return encoder.Encode(&serialized)
}