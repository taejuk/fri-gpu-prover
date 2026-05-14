typedef unsigned hyper field_element;

struct CompressInput {
    field_element left;
    field_element right;
};

struct CompressResult {
    field_element result;
};

program SIMPLE_POSEIDON_PROG {
    version SIMPLE_V1 {
        CompressResult poseidon_compress(CompressInput) = 1;
    } = 1;
} = 0x20001235;
