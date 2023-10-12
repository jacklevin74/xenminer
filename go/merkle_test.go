package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"testing"
)

func hashValue(value string) string {
	hash := sha256.Sum256([]byte(value))
	return hex.EncodeToString(hash[:])
}

func buildMerkleTree(elements []string, merkleTree map[string][2]string) string {
	if len(elements) == 1 {
		return elements[0]
	}

	var newElements []string
	for i := 0; i < len(elements); i += 2 {
		left := elements[i]
		var right string
		if i+1 < len(elements) {
			right = elements[i+1]
		} else {
			right = left
		}
		combined := left + right
		newHash := hashValue(combined)
		merkleTree[newHash] = [2]string{left, right}
		newElements = append(newElements, newHash)
	}
	return buildMerkleTree(newElements, merkleTree)
}

func TestHashValue(t *testing.T) {
	expected := sha256.Sum256([]byte("hello"))
	if hashValue("hello") != hex.EncodeToString(expected[:]) {
		t.Errorf("hashValue function not working properly")
	}
}

func TestBuildMerkleTree(t *testing.T) {
	elements := []string{"a", "b", "c", "d"}
	merkleTree := make(map[string][2]string)
	root := buildMerkleTree(elements, merkleTree)
	fmt.Printf("Merkle Root: %s\n", root)

	expectedRoot := hashValue(hashValue("a"+"b") + hashValue("c"+"d"))
	if root != expectedRoot {
		t.Errorf("buildMerkleTree function not working properly")
	}
	// Add more tests to validate the structure of merkleTree if necessary.
}

