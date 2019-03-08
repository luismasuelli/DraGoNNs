package cmd


import "../ffnn"


const Filename = "./network"


func NewMNISTNetwork() *ffnn.FFNetwork {
	networkBuilder := ffnn.New(0.01, 784, ffnn.HalfSquaredError{})
	networkBuilder.AddLayer(200, ffnn.Sigmoid{})
	networkBuilder.AddLayer(10, ffnn.Sigmoid{})
	return networkBuilder.Build()
}

func SaveMNISTNetwork(network *ffnn.FFNetwork) error {
	return ffnn.Save(network, Filename)
}

func LoadMNISTNetwork() (*ffnn.FFNetwork, error) {
	return ffnn.Load(Filename)
}
