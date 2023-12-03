package main

import (
        "bytes"
        "compress/zlib"
        "encoding/base64"
        "io"
        "log"
        "regexp"
        "strconv"

        "golang.org/x/crypto/argon2"
)


// decompressData decompresses a slice of bytes and returns the resulting string.
func decompressData(compressed []byte) (string, error) {
        b := bytes.NewReader(compressed)
        r, err := zlib.NewReader(b)
        if err != nil {
                return "", err
        }
        defer r.Close()

        var out bytes.Buffer
        if _, err = io.Copy(&out, r); err != nil {
                return "", err
        }
        return out.String(), nil
}

// verifyArgon2Hash checks the validity of an Argon2 hash given the hash and key.
func verifyArgon2Hash(hashToVerify, key string) bool {
        re := regexp.MustCompile(`^\$argon2id\$v=(\d+)\$m=(\d+),t=(\d+),p=(\d+)\$([A-Za-z0-9+/]+)\$([A-Za-z0-9+/]+)`)
        matches := re.FindStringSubmatch(hashToVerify)

        if len(matches) != 7 {
                log.Printf("Hash string is not in the correct format")
                return false
        }

        //version, _ := strconv.ParseUint(matches[1], 10, 32)
        memory, _ := strconv.ParseUint(matches[2], 10, 32)
        time, _ := strconv.ParseUint(matches[3], 10, 32)
        parallelism, _ := strconv.ParseUint(matches[4], 10, 8)
        salt, _ := base64.RawStdEncoding.DecodeString(matches[5])
        decodedHash, _ := base64.RawStdEncoding.DecodeString(matches[6])

        computedHash := argon2.IDKey([]byte(key), salt, uint32(time), uint32(memory), uint8(parallelism), uint32(len(decodedHash)))

        return bytes.Equal(computedHash, decodedHash)
}
