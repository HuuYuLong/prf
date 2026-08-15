# m is the number of neurons per layer
E_mac = 4.6  # pJ
E_mul = 3.7  # pJ
E_ac = 0.9  # pJ


def lif(m, T):
    return m * T * E_mac


def psn(m, T):
    return m * T * T * E_mac


def m_psn(m, T, k):
    return m * k * T * E_mac


def s_psn(m, T, k):
    return m * k * T * E_mac


def pmsn(m, T, n=4):
    return 8 * (n - 1) * m * T * E_mac


#
def adLIF(m, T):
    return 7 * m * T * E_mul + 4 * m * T * E_ac


def tcLIF(m, T):
    return 2 * m * T * E_mul + 5 * m * T * E_ac


def tsLIF(m, T):
    return 6 * m * T * E_mul + 6 * m * T * E_ac


def dhLIF(m, T, d=4):
    return 2 * d * m * T * E_mul + 2 * d * m * T * E_ac + d * m * T * E_mul + 2 * m * T * E_ac


def bhrf(m, T, ):
    return 6 * m * T * E_mul + 8 * m * T * E_ac


def elm(m, T, ):
    mlp_compute = 2 * (2 * m) * m * E_mac  # elm use 2 layer FC
    return (2 * m * m * E_mac + 4 * m * E_mul + 2 * m * E_ac + mlp_compute) * T


#
def prf(m, T):
    return 5 * m * T * E_mul + 3 * m * T * E_ac


if __name__ == '__main__':
    m = 128
    T = 2048
    k = 32  # for m_psn and s_psn

    print(f"lif energy: {lif(m, T) / 1e6} uJ")
    print("----------------------------------")
    print("Parallel Neuron")
    print(f"psn energy: {psn(m, T) / 1e6} uJ, ratio = {psn(m, T) / lif(m, T)}")
    print(f"m_psn energy: {m_psn(m, T, k) / 1e6} uJ, ratio = {m_psn(m, T, k) / lif(m, T)}")
    print(f"s_psn energy: {s_psn(m, T, k) / 1e6} uJ, ratio = {s_psn(m, T, k) / lif(m, T)}")
    print(f"pmsn energy: {pmsn(m, T) / 1e6} uJ, ratio = {pmsn(m, T) / lif(m, T)}")
    print("----------------------------------")
    print(f"adLIF energy: {adLIF(m, T) / 1e6} uJ, ratio = {adLIF(m, T) / lif(m, T)}")
    print(f"tcLIF energy: {tcLIF(m, T) / 1e6} uJ, ratio = {tcLIF(m, T) / lif(m, T)}")
    print(f"tsLIF energy: {tsLIF(m, T) / 1e6} uJ, ratio = {tsLIF(m, T) / lif(m, T)}")
    print(f"dhLIF energy: {dhLIF(m, T) / 1e6} uJ, ratio = {dhLIF(m, T) / lif(m, T)}")
    print(f"BHRF energy: {bhrf(m, T) / 1e6} uJ, ratio = {bhrf(m, T) / lif(m, T)}")
    print("----------------------------------")
    print(f"elm energy: {elm(m, T) / 1e6} uJ, ratio = {elm(m, T) / lif(m, T)}")
    print("----------------------------------")
    print(f"prf energy: {prf(m, T) / 1e6} uJ, ratio = {prf(m, T) / lif(m, T)}")
