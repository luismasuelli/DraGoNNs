package cmd

import (
	"../ffnn"
	"os"
	"encoding/csv"
	"bufio"
	"strconv"
	"gonum.org/v1/gonum/mat"
	"time"
	"fmt"
	"io"
)


const TrainingFile = "./mnist_train.csv"
const TestingFile = "./mnist_test.csv"


func makeInput(record []string) *mat.Dense {
	inputs := make([]float64, 784)
	for i := range inputs {
		x, _ := strconv.ParseFloat(record[i + 1], 64)
		inputs[i] = (x / 255.0 * 0.99) + 0.01
	}
	return mat.NewDense(784, 1, inputs)
}


func makeTarget(record []string) *mat.Dense {
	targets := make([]float64, 10)
	for i := range targets {
		targets[i] = 0.01
	}
	x, _ := strconv.Atoi(record[0])
	targets[x] = 0.99
	return mat.NewDense(10, 1, targets)
}


func makePair(record []string) (*mat.Dense, *mat.Dense) {
	return makeInput(record), makeTarget(record)
}


func TrainMNISTNetwork(network *ffnn.FFNetwork, epochs int) {
	fmt.Printf("Starting the training with %v epocs\n", epochs)
	t1 := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		if trainFile, err := os.Open(TrainingFile); err == nil {
			fmt.Println("Starting epoch:", epochs)
			csvReader := csv.NewReader(bufio.NewReader(trainFile))
			first := true
			index := 0
			for {
				var record []string
				var err error
				if record, err = csvReader.Read(); err != nil {
					break
				}

				if first {
					first = false
					continue
				}

				// train the NN with that data
				inputs, targets := makePair(record)
				network.Train(inputs, targets)
				index++
			}
			trainFile.Close()
			fmt.Println("Epoch ended.")
		} else {
			fmt.Printf("Epoch could not be started! : %v\n", err)
		}
	}
	elapsed := time.Since(t1)
	fmt.Printf("Training used %v epoch and took: %v\n", epochs, elapsed)
}


func TestMNISTNetwork(network *ffnn.FFNetwork) {
	t1 := time.Now()
	if testFile, err := os.Open(TestingFile); err == nil {
		fmt.Println("Starting test.")
		csvReader := csv.NewReader(bufio.NewReader(testFile))
		scores := make([]float64, 64)
		index := 0
		first := true
		for {
			record, err := csvReader.Read()
			if err == io.EOF {
				break
			}

			if first {
				first = false
				continue
			}

			inputs, expectedOutputs := makePair(record)
			outputs, cost := network.Test(inputs, expectedOutputs)

			// Get the highest output index
			highestOutputIndex := -1
			highestOutputValue := 0.0
			for index := 0; index < 10; index++ {
				value := outputs.At(index, 0)
				if value > highestOutputValue {
					highestOutputValue = value
					highestOutputIndex = index
				}
			}

			// Tailing the cost
			index++
			if index == 64 {
				index = 0
			}
			scores[index] = cost

			fmt.Printf("Case:\n  Expected: %v\n  Got: %v\n  Cost: %v\n", record[0], highestOutputIndex, cost)
		}
		elapsed := time.Since(t1)
		fmt.Printf("Test ended. Time taken to check: %s\n", elapsed)
		fmt.Printf("Last 64 scores: %v\n", scores)
		fmt.Println("Avg score of last 64 cases:", mat.Sum(mat.NewDense(1, 64, scores)) / 64)
	}
}