import torch

def show_sum_error(_range= range(-2**15,2**15)):

    L = [i for i in _range]

    I = torch.tensor(L, dtype=torch.int16)
    F16 = torch.tensor(L, dtype=torch.float16)
    F32 = torch.tensor(L, dtype=torch.float32)
    F64 = torch.tensor(L, dtype=torch.float64)

    m16 = torch.tensor(F16.numpy() , dtype = torch.float64)
    m32 = torch.tensor(F32.numpy() , dtype = torch.float64)
    m64 = torch.tensor(F64.numpy() , dtype = torch.float64)


    d16 = I-m16
    d32 = I-m32
    d64 = I-m64

    print("Conversion Errors from int16 in",_range,"maxed by (max is=) :")
    print("\tFloat 16 ", float(abs(d16).max()))
    print("\tFloat 32 ", float(abs(d32).max()))
    print("\tFloat 64 ", float(abs(d64).max()))