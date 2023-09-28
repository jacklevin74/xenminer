#ifndef ARGON2COMMON_H
#define ARGON2COMMON_H

namespace argon2 {

enum {
    ARGON2_BLOCK_SIZE = 1024,
    ARGON2_SYNC_POINTS = 4,
    ARGON2_PREHASH_DIGEST_LENGTH = 64,
    ARGON2_PREHASH_SEED_LENGTH = 72,
};

enum Type {
    ARGON2_D = 0,
    ARGON2_I = 1,
    ARGON2_ID = 2,
};

enum Version {
    ARGON2_VERSION_10 = 0x10,
    ARGON2_VERSION_13 = 0x13,
};

} // namespace argon2


#endif // ARGON2COMMON_H

