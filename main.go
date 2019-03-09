package main

import (
	"fmt"
	"time"
	"os"
	"math/rand"
	"bufio"
	"strings"
	"./cmd"
)


func trainNew() {
	network := cmd.NewMNISTNetwork()
	fmt.Println("Network created. Training...")
	cmd.TrainMNISTNetwork(network, 30)
	fmt.Println("Network trained. Saving...")
	if err := cmd.SaveMNISTNetwork(network); err != nil {
		fmt.Printf("Could not save the network! : %v\n", err)
	} else {
		fmt.Println("Network successfully saved.")
	}
}


func trainExisting() {
	if network, err := cmd.LoadMNISTNetwork(); err != nil {
		fmt.Printf("Could not load the network! : %v\n", err)
	} else {
		fmt.Println("Network loaded. Training...")
		cmd.TrainMNISTNetwork(network, 30)
		fmt.Println("Network trained. Saving...")
		if err := cmd.SaveMNISTNetwork(network); err != nil {
			fmt.Printf("Could not save the network! : %v\n", err)
		} else {
			fmt.Println("Network successfully saved.")
		}
	}
}


func testExisting() {
	if network, err := cmd.LoadMNISTNetwork(); err != nil {
		fmt.Printf("Could not load the network! : %v\n", err)
	} else {
		fmt.Println("Network loaded. Testing...")
		cmd.TestMNISTNetwork(network)
		fmt.Println("Network tested.")
	}
}


var keyboardInput = bufio.NewReader(os.Stdin)
func menu() {
	for {
		fmt.Print("Choose your option (train (n)ew, train (e)xisting, (t)est or (q)uit):")
		if result, err := keyboardInput.ReadString('\n'); err == nil {
			result = strings.TrimRight(result,"\n")
			switch result {
			case "n":
				trainNew()
			case "e":
				trainExisting()
			case "t":
				testExisting()
			case "q":
				fmt.Println("Have a nice day!")
				return
			}
		} else {
			fmt.Printf("Breaking in menu due to error : %v", err)
			return
		}
	}
}


func main() {
	// Prepare the random
	rand.Seed(time.Now().UTC().UnixNano())
	menu()
}