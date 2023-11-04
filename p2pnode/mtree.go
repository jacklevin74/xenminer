package main

import (
	"crypto/sha256"
	"encoding/hex"
)

type MerkleNode struct {
	Left  string
	Right string
}

func hashValue(data string) string {
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

func buildMerkleTree(elements []string, merkleTree map[string]MerkleNode) (string, map[string]MerkleNode) {
	if len(elements) == 1 {
		return elements[0], merkleTree
	}

	var newElements []string
	for i := 0; i < len(elements); i += 2 {
		left := elements[i]
		right := left
		if i+1 < len(elements) {
			right = elements[i+1]
		}

		combined := left + right
		newHash := hashValue(combined)
		merkleTree[newHash] = MerkleNode{Left: left, Right: right}
		newElements = append(newElements, newHash)
	}
	return buildMerkleTree(newElements, merkleTree)
}
