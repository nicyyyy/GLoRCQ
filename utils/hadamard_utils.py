# import torch, math
# import fast_hadamard_transform
# import numpy as np

# def walsh_matrix(k, device, normalized=True):
#     n = 2 ** k
#     H = np.ones((1, 1), dtype=int)
#     for i in range(k):
#         H = np.kron(H, np.array([[1, 1], [1, -1]]))
#     if normalized:
#         H = H / np.sqrt(n)
    
#     gray = [i ^ (i >> 1) for i in range(n)]
#     order = sorted(range(n), key=lambda i: gray[i])
#     W = H[order, :]
#     W = W[:, order]
#     return torch.tensor(W, dtype=torch.float64).to(device)

# def block_diagonal_walsh_matrix(m, n, device, normalized=True):

#     if m % n != 0:
#         raise ValueError("m must integer times of n!")
#     if n & (n - 1) != 0 or n < 1:
#         raise ValueError("n must power of 2!")

#     k_blocks = m // n

#     W_block_torch = walsh_matrix(int(np.log2(n)), device, normalized=normalized)
    
#     Q = torch.zeros(m, m, dtype=torch.float64)
#     for i in range(k_blocks):
#         start = i * n
#         end = start + n
#         Q[start:end, start:end] = W_block_torch
#     Q = Q.to(device)
#     return Q

# def random_hadamard_matrix(size, device):
#     # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
#     Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
#     Q = Q * 2 - 1
#     Q = torch.diag(Q)
#     return matmul_hadU(Q).to(device)


# def create_diagI_matrix_upper(original_matrix, n):
#     m = original_matrix.size(0)
#     assert m >= n, "m mast largger n"
#     assert original_matrix.dim() == 2 and original_matrix.size(0) == original_matrix.size(1), "not unit matrix!"
    
#     result = torch.zeros(m, m, dtype=original_matrix.dtype, device=original_matrix.device)
#     if m > n:
#         result[:m-n, :m-n] = torch.eye(m-n, dtype=original_matrix.dtype, device=original_matrix.device)
#     result[m-n:, m-n:] = original_matrix[m-n:, m-n:]
    
#     return result

# def create_diagI_matrix_lower(original_matrix, n):
#     m = original_matrix.size(0)
#     assert m >= n, "m must be larger than or equal to n"
#     assert original_matrix.dim() == 2 and original_matrix.size(0) == original_matrix.size(1), "Matrix must be square"

#     result = torch.zeros(m, m, dtype=original_matrix.dtype, device=original_matrix.device)

#     if n > 0:
#         result[:n, :n] = original_matrix[:n, :n]
#     if m - n > 0:
#         result[n:, n:] = torch.eye(m - n, dtype=original_matrix.dtype, device=original_matrix.device)
    
#     return result


# def construct_partial_permutation_matrix_upper(S, m=512, dtype = torch.float16):
#     S = np.array(S.cpu())
#     all_indices = set(range(m))
#     S_set = set(S)
#     front = list(S)
#     back = sorted(all_indices - S_set)
    
#     sigma = front+back

#     P = torch.zeros((m, m), dtype=dtype)
#     for j, col_idx in enumerate(sigma):
#         P[col_idx, j] = 1.0
    
#     return P


# def construct_partial_permutation_matrix_lower(S, m=512, dtype = torch.float16):
#     S = np.array(S.cpu())
#     all_indices = set(range(m))
#     S_set = set(S)
#     front = list(S)
#     back = sorted(all_indices - S_set)
    
#     sigma = back + front

#     P = torch.zeros((m, m), dtype=dtype)
#     for j, col_idx in enumerate(sigma):
#         P[col_idx, j] = 1.0
    
#     return P


