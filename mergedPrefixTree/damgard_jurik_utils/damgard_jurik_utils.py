import damgard_jurik_local
from random import randint
import pickle
from gmpy2 import mpz
from damgard_jurik_local.crypto import keygen, PrivateKeyShare

import damgard_jurik_local.crypto

public_key, private_key_shares = keygen(
    n_bits=64,
    s=1,
    threshold=2,
    n_shares=4
)
print("keys generated")

private_key_1: PrivateKeyShare = private_key_shares[0]
private_key_2: PrivateKeyShare = private_key_shares[1]

c1 = public_key.encrypt(1)
c2 = public_key.encrypt(0)

print(c1.value)


c_list = [private_key_1.decrypt(c1), private_key_2.decrypt(c1)]
i_list = [private_key_1.i, private_key_2.i]
print(private_key_1.final_decrypt(c_list, i_list))


# m_1, m_2 = 421231246423747862364768234, 33
# c_1, c_2 = public_key.encrypt(m_1), public_key.encrypt(m_2)
# print(pickle.dumps(c_1.value))
# print("ciphertexts encrypted")
# c = c_1 + c_2
# print("ciperthexts combined")
# m_prime = private_key_ring.decrypt(c)
# print("ciphertext decrypted")
# print(m_prime)