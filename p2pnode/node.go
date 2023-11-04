package main

import (
	"slices"
)

const (
	SupernodeRole = "supernode"
	RelayRole     = "relay"
	MinerRole     = "miner"
	ValidatorRole = "validator"
	BootstrapRole = "bootstrap"
	RpcRole       = "rpc"
)

var supportedRoles = []string{
	SupernodeRole,
	RelayRole,
	MinerRole,
	ValidatorRole,
	BootstrapRole,
	RpcRole,
}

type Node struct {
	roles []string
}

func (n Node) isSupernode() bool {
	return slices.Contains(n.roles, SupernodeRole)
}

func (n Node) isRelay() bool {
	return slices.Contains(n.roles, RelayRole)
}

func (n Node) isMiner() bool {
	return slices.Contains(n.roles, MinerRole)
}

func (n Node) isValidator() bool {
	return slices.Contains(n.roles, ValidatorRole)
}

func (n Node) isBootstrap() bool {
	return slices.Contains(n.roles, BootstrapRole)
}

func (n Node) isRpc() bool {
	return slices.Contains(n.roles, RpcRole)
}
