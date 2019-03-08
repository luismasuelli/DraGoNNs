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


func TrainMNISTNetwork(network *ffnn.FFNetwork) {
	fmt.Println("Starting the training.")
	t1 := time.Now()
	for epochs := 0; epochs < 5; epochs++ {
		if trainFile, err := os.Open(TrainingFile); err == nil {
			fmt.Println("Starting epoch.")
			csvReader := csv.NewReader(bufio.NewReader(trainFile))
			for {
				var record []string
				var err error
				if record, err = csvReader.Read(); err != nil {
					break
				}

				// train the NN with that data
				network.Train(makePair(record))
			}
			trainFile.Close()
			fmt.Println("Epoch ended.")
		} else {
			fmt.Printf("Epoch could not be started! : %v\n", err)
		}
	}
	elapsed := time.Since(t1)
	fmt.Printf("Training used 5 epochs and took: %s\n", elapsed)
}


func TestMNISTNetwork(network *ffnn.FFNetwork) {
	t1 := time.Now()
	if testFile, err := os.Open(TestingFile); err == nil {
		fmt.Println("Starting test.")
		csvReader := csv.NewReader(bufio.NewReader(testFile))
		scores := make([]float64, 64)
		index := 0
		for {
			record, err := csvReader.Read()
			if err == io.EOF {
				break
			}

			inputs, expectedOutputs := makePair(record)
			_, cost := network.Test(inputs, expectedOutputs)
			// Tailing the cost
			index++
			if index == 64 {
				index = 0
			}
			scores[index] = cost

			// Displaying the output and cost
			// fmt.Printf("Case:\n  Expected: %v\n  Output %v\n  Cost: %v\n", expectedOutputs, outputs, cost)
		}
		elapsed := time.Since(t1)
		fmt.Printf("Test ended. Time taken to check: %s\n", elapsed)
		fmt.Printf("Last 64 scores: %v\n", scores)
		fmt.Println("Avg score of last 64 cases:", mat.Sum(mat.NewDense(1, 64, scores)) / 64)
	}
}