import numpy as np 
from math import gcd 
from fractions import Fraction 
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from dotenv import load_dotenv
import os

print("Quantum module loaded")

load_dotenv()

token = os.getenv("IBMQ_TOKEN")
print("Token loaded")
print(token)

service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=token,
        )

print("qiskit runtime service loaded")

def find_valid_a(N):
    valid_a = []

    def order_mod(a, n):
        if gcd(a, n) != 1:
            return 0 
        for i in range(1, n):
            if pow(a, i, n) == 1:
                return i 
        return 0 

    for a in range(2, N):
        if gcd(a, N) == 1:
            order = order_mod(a, N)
            if order > 2:
                valid_a.append(a)
        
    return valid_a

N = 77
a = 5  # co-prime with N
N_COUNT = 15  # number of counting qubits (77 = 1001101 => 7 bits (2*7+1 = 15))

valid_a = find_valid_a(N)
print(f"Found {len(valid_a)} valid 'a's: {valid_a}")
print(valid_a)


# Create Cuccaro adder
def qadd_in_place(qc, x_reg, y_reg, ancillas):
    def majority(qc, a, b, c):
        qc.cx(c, b)
        qc.cx(c, a)
        qc.ccx(a, b, c)

    def unmaj(qc, a, b, c):
        qc.ccx(a, b, c)
        qc.cx(c, a)
        qc.cx(a, b)

    for i in range(6):
        majority(qc, x_reg[i], y_reg[i], y_reg[i+1])

    qc.cx(x_reg[6], y_reg[6])

    for i in range(5, -1, -1):
        unmaj(qc, x_reg[i], y_reg[i], y_reg[i+1])

def qcompare_77(qc, target_reg, ancilla_flag, ancillas):
    qc.ccx(ancilla_flag, ancillas[0], target_reg[1])
    qc.ccx(ancilla_flag, ancillas[0], target_reg[2])
    qc.ccx(ancilla_flag, ancillas[0], target_reg[5])

    qc.cx(target_reg[6], ancillas[1])

    for i in range(5, -1, -1):
        if (77 >> i) & 1:
            qc.x(target_reg[i])
            qc.ccx(ancillas[1], target_reg[i], ancillas[2])
            qc.x(target_reg[i])
        else:
            qc.ccx(ancillas[1], target_reg[i], ancillas[2])

        qc.cx(ancillas[2], ancillas[1])
        qc.x(ancillas[2])

    qc.cx(ancillas[1], ancilla_flag)

    for i in range(0, 6):
        if(77 >> i) & 1:
            qc.x(target_reg[i])
            qc.ccx(ancillas[1], target_reg[i], ancillas[2])
            qc.x(target_reg[i])
        else:
            qc.ccx(ancillas[1], target_reg[i], ancillas[2])
        qc.cx(ancillas[2], ancillas[1])
        qc.x(ancillas[2])

    qc.cx(target_reg[6], ancillas[1])

def qsub_77_if_flag(qc, target_reg, ancilla_flag, ancillas):
    qc.ccx(ancilla_flag, ancillas[0], target_reg[1])
    qc.ccx(ancilla_flag, ancillas[0], target_reg[2])
    qc.ccx(ancilla_flag, ancillas[0], target_reg[5])

    qcompare_77(qc, target_reg, ancilla_flag, ancillas)

def mod_reduce_77(qc, target_reg, ancilla_flag, ancillas):
    qcompare_77(qc, target_reg, ancilla_flag, ancillas)
    qsub_77_if_flag(qc, target_reg, ancilla_flag, ancillas)

def multiply_by_a_mod_77(qc, x_reg, accumulator_reg, ancilla_flag, ancillas, a):
    a_binaryform = bin(a)[2:].zfill(7)
    print(f"Multiplying by {a} (binary: {a_binaryform})")


    for i, bit in enumerate(reversed(a_binaryform)):
        if bit == '1':
            qadd_in_place(qc, accumulator_reg, x_reg, ancillas)
            mod_reduce_77(qc, accumulator_reg, ancilla_flag, ancillas)

def c_amod77(a, power):
    print(f"Creating controlled gate for {a}^{power} mod 77")
    if a not in valid_a:
        raise ValueError("'a' must be chosen from the list of valid 'a's")
    
    # Create registers
    x_reg = QuantumRegister(7, 'x')
    accumulator_reg = QuantumRegister(7, 'acc')
    ancilla_flag = QuantumRegister(1, 'flag')
    ancillas = QuantumRegister(3, 'anc')
    
    # Create circuit with all necessary registers
    qc = QuantumCircuit(x_reg, accumulator_reg, ancilla_flag, ancillas)

    #optimalization to create circuit by power not multiplying (a^x vs a*a*a*a*a*a)
    binary_power = bin(power)[2:]
    current_value = a

    print(f" Using square-and-multiply for power {power}")
    for bit in binary_power:
        if bit == '1':
            multiply_by_a_mod_77(qc, x_reg, accumulator_reg, ancilla_flag[0], ancillas, current_value)
            current_value = (current_value * current_value) % 77
    
    gate = qc.to_gate()
    gate.name = f"{a}^{power} mod 77"
    return gate.control(1)

def qft_dagger(n):
    qc = QuantumCircuit(n)
    # Swap qubits
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    # Apply controlled-phase and Hadamard
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / 2**(j - m), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

# Create main circuit
print("Creating main circuit...")
counting_qubits = QuantumRegister(N_COUNT, 'counting')
x_reg = QuantumRegister(7, 'x')
accumulator = QuantumRegister(7, 'acc')
flag = QuantumRegister(1, 'flag')
ancillas = QuantumRegister(3, 'anc')
c = ClassicalRegister(N_COUNT, 'c')
print("Registers created.")

qc = QuantumCircuit(counting_qubits, x_reg, accumulator, flag, ancillas, c)
print("Starting Hadamard gates...")

# Initialize counting qubits in superposition
for i in range(N_COUNT):
    qc.h(i)

print("Starting modular multiplication...")

# Set x_reg to |1>
qc.x(N_COUNT)

# Apply controlled modular multiplication
for i in range(N_COUNT):
    qc.append(c_amod77(a, 2**i), 
              [i] + list(range(N_COUNT, N_COUNT + 18)))  # +18 accounts for all auxiliary qubits

print("Circuit construction completed")

# Apply inverse QFT to counting qubits
qc.append(qft_dagger(N_COUNT), range(N_COUNT))

# Measure counting qubits
qc.measure(range(N_COUNT), range(N_COUNT))
qc.name = "Shor_Example"

# Execute the circuit
backend = service.least_busy(simulator=False)
print("Using backend:", backend.name)
transpiled_circ = transpile(qc, backend=backend)
sampler = Sampler(mode=backend)
print("Sampler is ready.")

job = sampler.run([transpiled_circ], shots=4096)
print(f"Submitted job. Job ID: {job.job_id()}")
result = job.result()
pub_result = result[0]  # The single-circuit result
counts_dict = pub_result.data.c.get_counts()  # 'c' is the auto-named classical register
shots_used = sum(counts_dict.values())
counts_prob = {bitstring: c / shots_used for bitstring, c in counts_dict.items()}

print("\n===== Measurement Results (Top 10) =====")
sorted_counts = sorted(counts_prob.items(), key=lambda x: x[1], reverse=True)[:10]
for (bitstring, prob) in sorted_counts:
    print(f"  {bitstring}: p={prob:.4f}")

phases = []
for bitstring, prob in counts_prob.items():
    if prob > 0:
        decimal = int(bitstring, 2)
        phase = decimal / (2**N_COUNT)
        phases.append((bitstring, phase, prob))

phases.sort(key=lambda x: x[2], reverse=True)
best_bitstring, best_phase, best_prob = phases[0]
print(f"\nMost likely outcome: {best_bitstring} with p={best_prob:.4f}")
print(f"Corresponding phase (decimal/2^N_COUNT) = {best_phase}")

frac = Fraction(best_phase).limit_denominator(N)
r = frac.denominator
s = frac.numerator
print(f"Fraction from phase = {s}/{r}")

guess1 = gcd(a**(r//2) - 1, N)
guess2 = gcd(a**(r//2) + 1, N)
print("\nPotential factors from guess:")
print(f"  gcd({a}^({r//2}) - 1, {N}) = {guess1}")
print(f"  gcd({a}^({r//2}) + 1, {N}) = {guess2}")
print("\nDone!")
