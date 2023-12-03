// SPDX-License-Identifier: MIT
pragma solidity ^0.7.0;
pragma abicoder v2;  // Enable ABI coder v2

contract AddressToStringMapping {
    struct Record {
        uint256 blockId;
        int mValue;
        int tValue;
        int pValue;
        bytes saltBytes;
        bytes key;
        uint8 currencyType;
        uint256 amount;
    }

    mapping(address => mapping(uint256 => Record)) private data;
    mapping(address => uint256) private recordCount;

    function setData(address _address, uint256 _blockId, int _mValue, int _tValue, int _pValue, bytes memory _saltBytes, bytes memory _key, uint8 _currencyType, uint256 _amount) public {
        Record memory newRecord = Record({
            blockId: _blockId,
            mValue: _mValue,
            tValue: _tValue,
            pValue: _pValue,
            saltBytes: _saltBytes,
            key: _key,
            currencyType: _currencyType,
            amount: _amount
        });

        data[_address][recordCount[_address]] = newRecord;
        recordCount[_address]++;
    }

    function getData(address _address, uint256 _index) public view returns (Record memory) {
        return data[_address][_index];
    }

    function getRecordCount(address _address) public view returns (uint256) {
        return recordCount[_address];
    }
}
