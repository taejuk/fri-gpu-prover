typedef unsigned hyper field_element;

const MAX_COEFFS = 1048576;

struct PolyInput {
    unsigned int log_n;
    field_element coeffs<MAX_COEFFS>;
};

struct FriResult {
    field_element root;
};

program FRI_RPC_PROG {
    version FRI_V1 {
        FriResult compute_fri_root(PolyInput) = 1;
    } = 1;
} = 0x20001234;


