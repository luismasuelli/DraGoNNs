package ffnn

import (
	"os"
	"encoding/json"
	"errors"
	"strings"
	"gonum.org/v1/gonum/mat"
)


type serializedFFLayer struct {
	Activator           string
	OutputSize          int
	WeightsData         []byte
}
type serializedFFNetwork struct {
	ErrorMetric         string
	DefaultLearningRate float64
	InputSize           int
	Layers              []*serializedFFLayer
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

	if serialized.InputSize < 1 {
		return nil, errors.New("input size must be >= 1")
	}

	layersCount := len(serialized.Layers)
	if len(serialized.Layers) == 0 {
		return nil, errors.New("at least one layer must be present")
	}

	if serialized.DefaultLearningRate <= 0 {
		return nil, errors.New("learning rate must be positive (and, preferably, small)")
	}

	network := &FFNetwork{
		defaultLearningRate: serialized.DefaultLearningRate,
		errorMetric: GetErrorMetric(serialized.ErrorMetric),
		layers: make([]*FFLayer, layersCount),
		activatorDerivativeResultsOverWeightedInputs: make([]*mat.Dense, layersCount),
		activationsCostGradients: make([]*mat.Dense, layersCount),
		errorsOverWeightedInputs: make([]*mat.Dense, layersCount),
	}

	inputSize := serialized.InputSize
	for index, serializedLayer := range serialized.Layers {
		outputSize := serializedLayer.OutputSize
		if outputSize < 1 {
			return nil, errors.New("output size must be >= 1")
		}
		activator := GetActivator(serializedLayer.Activator)

		if layer, err := loadLayer(inputSize, outputSize, activator, serializedLayer.WeightsData); err != nil {
			return nil, err
		} else {
			network.layers[index] = layer
			// here we create the training matrices
			network.activatorDerivativeResultsOverWeightedInputs[index] = mat.NewDense(outputSize, 1, nil)
			network.activationsCostGradients[index] = mat.NewDense(outputSize, 1, nil)
			network.errorsOverWeightedInputs[index] = mat.NewDense(outputSize, 1, nil)
		}

		// output size is the new input size
		inputSize = serializedLayer.OutputSize
	}

	return network, nil
}


func Save(network *FFNetwork, filename string) (error) {
	filename = withExtension(filename, "ffnn")
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
		DefaultLearningRate: network.defaultLearningRate,
		InputSize: network.layers[0].inputSize,
		Layers: make([]*serializedFFLayer, len(network.layers)),
		ErrorMetric: network.errorMetric.Name(),
	}
	for index, layer := range network.layers {
		if weightsData, err := encodeFFLayer(layer); err != nil {
			return err
		} else {
			serialized.Layers[index] = &serializedFFLayer{
				Activator: layer.activator.Name(),
				OutputSize: layer.outputSize,
				WeightsData: weightsData,
			}
		}
	}

	encoder := json.NewEncoder(file)
	return encoder.Encode(&serialized)
}


type FFLayerSpec struct {
	outputSize int
	activator Activator
}
type FFNetworkBuilder struct {
	defaultLearningRate float64
	inputSize int
	errorMetric ErrorMetric
	layers []*FFLayerSpec
}


func New(defaultLearningRate float64, inputSize int, errorMetric ErrorMetric) *FFNetworkBuilder {
	if inputSize < 1 {
		panic("input size must be >= 1")
	}

	if defaultLearningRate <= 0 {
		panic("learning rate must be positive (and, preferably, small)")
	}

	if errorMetric == nil {
		errorMetric = GetErrorMetric("_default")
	}

	return &FFNetworkBuilder{
		inputSize: inputSize,
		defaultLearningRate: defaultLearningRate,
		errorMetric: errorMetric,
		layers: make([]*FFLayerSpec, 0),
	}
}


func (builder *FFNetworkBuilder) AddLayer(outputSize int, activator Activator) *FFNetworkBuilder {
	if outputSize < 1 {
		panic("output size must be >= 1")
	}

	if activator == nil {
		activator = GetActivator("_default")
	}

	builder.layers = append(builder.layers, &FFLayerSpec{
		outputSize: outputSize,
		activator: activator,
	})
	return builder
}


func (builder *FFNetworkBuilder) CanBuild() bool {
	return len(builder.layers) > 0
}


func (builder *FFNetworkBuilder) Build() *FFNetwork {
	layersCount := len(builder.layers)
	if layersCount == 0 {
		panic("this builder must specify at least one layer")
	}

	network := &FFNetwork{
		defaultLearningRate: builder.defaultLearningRate,
		errorMetric: builder.errorMetric,
		layers: make([]*FFLayer, layersCount),
		activatorDerivativeResultsOverWeightedInputs: make([]*mat.Dense, layersCount),
		activationsCostGradients: make([]*mat.Dense, layersCount),
		errorsOverWeightedInputs: make([]*mat.Dense, layersCount),
	}

	inputSize := builder.inputSize
	for index, layerSpec := range builder.layers {
		network.layers[index] = newFFLayer(inputSize, layerSpec.outputSize, layerSpec.activator)
		// here we create the training matrices
		network.activatorDerivativeResultsOverWeightedInputs[index] = mat.NewDense(layerSpec.outputSize, 1, nil)
		network.activationsCostGradients[index] = mat.NewDense(layerSpec.outputSize, 1, nil)
		network.errorsOverWeightedInputs[index] = mat.NewDense(layerSpec.outputSize, 1, nil)
		inputSize = layerSpec.outputSize
	}

	return network
}
