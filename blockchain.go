package main

import (

	// "encoding/binary"
	"bytes"
	"crypto/sha256"
	"strconv"
	"time"
)

type Blockchain struct { //区块链
	Blocks []*Block
}

type Block struct {
	Timestamp     int64
	PrevBlockHash []byte
	Hash          []byte
	Iteration     int
	GlobalW       []float64
	Data          []Update
}

type Update struct {
	SourceID      int
	Iteration     int
	Delta         []float64 //梯度
	Commitment    int       // a commitment to delta:
	Noise         []float64
	Accepted      bool
	SignatureList []int //
}

func (b *Block) SetHash() {
	timestamp := []byte(strconv.FormatInt(b.Timestamp, 10)) // ：=[]  ？
	headers := bytes.Join([][]byte{b.PrevBlockHash, timestamp}, []byte{})
	hash := sha256.Sum256(headers)
	b.Hash = hash[:]
}

func NewBlock(iteration int, globalW []float64, data []Update, prevBlockHash []byte) *Block {

	var blockTime int64

	if len(data) == 0 {
		blockTime = 0
	} else {
		blockTime = time.Now().Unix()
	}

	block := &Block{blockTime, prevBlockHash, []byte{}, iteration, globalW, data}
	block.SetHash()

	return block
}

func (bc *Blockchain) AddBlock(iteration int, globalW []float64, data []Update) {
	prevBlock := bc.Blocks[len(bc.Blocks)-1]
	newBlock := NewBlock(iteration, globalW, data, prevBlock.Hash)
	bc.Blocks = append(bc.Blocks, newBlock)
}

func (bc *Blockchain) getLatestBlock() *Block {
	return bc.Blocks[len(bc.Blocks)-1]
}

func (bc *Blockchain) getLatestGradient() []float64 {

	prevBlock := bc.Blocks[len(bc.Blocks)-1]
	gradient := make([]float64, len(prevBlock.GlobalW))
	copy(gradient, prevBlock.GlobalW)
	return gradient
}

func (bc *Blockchain) verifyBlock(block Block) bool {

	return true
}
