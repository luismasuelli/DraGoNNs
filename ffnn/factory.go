package ffnn

import (
	"os"
	"encoding/json"
	"errors"
	"strings"
	"gonum.org/v1/gonum/mat"
	"fmt"
)


type serializedFFLayer struct {
	activator           string
	outputSize          int
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
func withoutExtension(filename string, extension string) string {
	if strings.Trim(filename, " \r\n\t") == "" {
		return ""
	}

	if strings.HasSuffix(filename, "." + extension) {
		filename += strings.TrimSuffix(filename, "." + extension)
	}
	return filename
}
func loadLayer(index int, inputSize int, outputSize int, activator Activator, baseFilename string) (*FFLayer, error) {
	// Open file for reading.
	// For a filename being /path/to/filename.ffnn, this one will be
	var file *os.File
	var err error
	var subName = withoutExtension(baseFilename, "ffnn") + fmt.Sprintf("-%v.fflayer", index)
	if file, err = os.Open(subName); err != nil {
		return nil, err
	} else {
		defer file.Close()
	}

	// Read everything
	return loadFFLayer(inputSize, outputSize, activator, file)
}


func Load(filename string) (*FFNetwork, error) {
	// Open file for reading
	filename = withExtension(filename, "ffnn")
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

		if layer, err := loadLayer(index, inputSize, outputSize, activator, filename); err != nil {
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